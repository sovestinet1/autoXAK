"""Real v2 recording-to-model integration, isolated from the legacy application.

Run with ``python -m unittest discover -s tests -p test_recording_model.py -v``.
Only an in-memory SQLite database is opened; app.main is never imported.
"""

import copy
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.dependencies import require_user
from app.api.recordings import router
from app.core.database import Base, get_db
from app.models.db_models import Trip, User
from app.models.recordings import RecordingChunk, TripRecording
from app.services.fuel_prediction import predict_fuel


START = 1_760_000_000_000
PROFILE = {
    "vehicle_id": "confirmed-pilot-car",
    "make": "Pilot",
    "model": "Confirmed modification",
    "fuel_type": "gasoline",
    "mass_kg": 1500,
    "displacement_l": 2.0,
    "turbo": True,
}


def gps_trace():
    # This trace has independently generated golden predictions in the original
    # saved-model experiment. It covers two 30 s windows and a 5 s tail.
    points = []
    for second in range(66):
        speed = second * 0.4 if second <= 30 else 12.0 if second <= 35 else 12.0 - (second - 35) * 0.2
        points.append({
            "timestamp_ms": START + second * 1000,
            "latitude": 55.0,
            "longitude": 37.0,
            "speed_mps": speed,
            "accuracy_m": 8.0,
            "speed_source": "gps",
        })
    return points


class RecordingModelTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with self.sessions() as session:
            self.user = User(
                id="model-test-owner", is_guest=True,
                total_savings_rub=123.45, total_fuel_saved_liters=1.5,
                total_distance_km=42.0, total_trips=3,
            )
            session.add(self.user)
            await session.commit()

        async def isolated_database():
            async with self.sessions() as session:
                yield session

        async def isolated_owner():
            return self.user

        application = FastAPI()
        application.include_router(router)
        application.dependency_overrides[get_db] = isolated_database
        application.dependency_overrides[require_user] = isolated_owner
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://model-test",
        )
        self.points = gps_trace()
        self.motion = [{
            "timestamp_ms": START + 500,
            "event_timestamp_ms": 500.0,
            "ax": 0.1, "ay": 0.2, "az": 9.8,
            "gx": 1.0, "gy": 2.0, "gz": 3.0,
            "acceleration_includes_gravity": True,
            "interval_ms": 20.0,
        }]
        self.completion = {
            "ended_at_ms": START + 65_000, "chunk_count": 1, "end_reason": "user",
        }

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.engine.dispose()

    async def upload(self, profile=None):
        response = await self.client.post("/api/v2/trips", json={
            "trip_id": "trip-model", "started_at_ms": START,
            "vehicle": copy.deepcopy(PROFILE if profile is None else profile),
        })
        self.assertEqual(response.status_code, 200, response.text)
        response = await self.client.put("/api/v2/trips/trip-model/chunks/0", json={
            "gps": self.points, "motion": self.motion,
        })
        self.assertEqual(response.status_code, 200, response.text)

    async def assert_legacy_totals_unchanged(self):
        async with self.sessions() as session:
            user = await session.get(User, self.user.id)
            self.assertEqual(user.total_savings_rub, 123.45)
            self.assertEqual(user.total_fuel_saved_liters, 1.5)
            self.assertEqual(user.total_distance_km, 42.0)
            self.assertEqual(user.total_trips, 3)
            self.assertEqual(await session.scalar(select(func.count()).select_from(Trip)), 0)

    async def test_real_model_is_saved_and_completion_retry_uses_cached_result(self):
        await self.upload()
        # A later garage edit must not change the car snapshot for this recording.
        response = await self.client.put("/api/v2/vehicle", json={**PROFILE, "mass_kg": 1900})
        self.assertEqual(response.status_code, 200, response.text)
        control = {
            "distance_km": 0.51, "average_l_per_100km": 50.0,
            "source": "trip_computer", "same_interval": True,
        }
        completion = {**self.completion, "control": control}
        with patch("app.services.fuel_prediction.predict_fuel", wraps=predict_fuel) as prediction:
            first = await self.client.post("/api/v2/trips/trip-model/complete", json=completion)
            retry = await self.client.post("/api/v2/trips/trip-model/complete", json=completion)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(retry.status_code, 200, retry.text)
        self.assertEqual(first.json(), retry.json())
        self.assertEqual(prediction.call_count, 1)
        arguments = prediction.call_args.args
        self.assertEqual(arguments[0], self.points)
        self.assertEqual(arguments[1]["mass_kg"], 1500)
        self.assertEqual(arguments[2:], (START, START + 65_000))
        result = first.json()
        estimate = result["fuel_estimate"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["control"], control)
        self.assertEqual(result["savings_status"], "unavailable")
        self.assertEqual(estimate["status"], "research_estimate")
        self.assertAlmostEqual(estimate["estimated_fuel_l"], 0.045101829846272055, places=12)
        self.assertEqual(estimate["complete_trip_fuel_l"], estimate["estimated_fuel_l"])
        self.assertEqual(estimate["coverage_fraction"], 1.0)
        self.assertNotIn("fuel_saved_liters", estimate)
        async with self.sessions() as session:
            stored = await session.get(TripRecording, "trip-model")
            self.assertEqual(stored.fuel_estimate, estimate)
            self.assertEqual(stored.control, control)
            self.assertEqual(await session.scalar(select(func.count()).select_from(RecordingChunk)), 1)
        await self.assert_legacy_totals_unchanged()

    async def test_unconfirmed_car_profile_keeps_raw_data_and_returns_null(self):
        await self.upload({**PROFILE, "fuel_type": "unknown", "turbo": None})
        with patch("app.services.fuel_prediction._load_artifacts") as load_artifacts:
            response = await self.client.post("/api/v2/trips/trip-model/complete", json=self.completion)
        self.assertEqual(response.status_code, 200, response.text)
        load_artifacts.assert_not_called()
        result = response.json()
        estimate = result["fuel_estimate"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["savings_status"], "unavailable")
        self.assertEqual(estimate["status"], "profile_incomplete")
        self.assertEqual(estimate["missing_profile_fields"], ["fuel_type", "turbo"])
        self.assertIsNone(estimate["estimated_fuel_l"])
        self.assertIsNone(estimate["complete_trip_fuel_l"])
        self.assertTrue(all(value is None for value in estimate["fuel_l_on_retained_intervals"].values()))
        exported = await self.client.get("/api/v2/trips/trip-model/export")
        self.assertEqual(exported.status_code, 200, exported.text)
        self.assertEqual(exported.json()["gps"], self.points)
        self.assertEqual(exported.json()["motion"], self.motion)
        await self.assert_legacy_totals_unchanged()

    async def test_model_exception_does_not_lose_raw_recording_or_complete_twice(self):
        await self.upload()
        with patch("app.services.fuel_prediction.predict_fuel", side_effect=RuntimeError("test failure")) as prediction:
            with self.assertLogs("app.api.recordings", level="ERROR"):
                first = await self.client.post("/api/v2/trips/trip-model/complete", json=self.completion)
            retry = await self.client.post("/api/v2/trips/trip-model/complete", json=self.completion)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(retry.status_code, 200, retry.text)
        self.assertEqual(prediction.call_count, 1)
        self.assertEqual(first.json(), retry.json())
        result = first.json()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["fuel_estimate"]["status"], "model_unavailable")
        self.assertIsNone(result["fuel_estimate"]["estimated_fuel_l"])
        self.assertIsNone(result["fuel_estimate"]["complete_trip_fuel_l"])
        self.assertIn("prediction_failed", result["fuel_estimate"]["quality_flags"])
        exported = await self.client.get("/api/v2/trips/trip-model/export")
        self.assertEqual(exported.status_code, 200, exported.text)
        self.assertEqual(exported.json()["gps"], self.points)
        self.assertEqual(exported.json()["motion"], self.motion)
        async with self.sessions() as session:
            stored = await session.get(TripRecording, "trip-model")
            self.assertEqual(stored.fuel_estimate, result["fuel_estimate"])
            self.assertEqual(stored.chunk_count, 1)
        await self.assert_legacy_totals_unchanged()


if __name__ == "__main__":
    unittest.main()
