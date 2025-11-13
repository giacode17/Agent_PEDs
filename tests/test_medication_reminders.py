# tests/test_medication_reminders.py
import pytest
import time
from peds_post_discharge_agent.medication_reminders import (
    MedicationReminderManager,
    MedicationSchedule
)


class TestMedicationReminderManager:
    """Test suite for medication reminder functionality."""

    def setup_method(self):
        """Create a fresh manager for each test."""
        self.manager = MedicationReminderManager()

    def teardown_method(self):
        """Clean up after each test."""
        self.manager.cancel_all_schedules()

    def test_parse_simple_medication(self):
        """Test parsing a simple medication schedule."""
        schedule = self.manager.parse_medication_input("Take Zyrtec every 12 hours")

        assert schedule is not None
        assert schedule.medication_name == "Zyrtec"
        assert schedule.interval_hours == 12.0
        assert schedule.duration_days is None

    def test_parse_medication_with_duration(self):
        """Test parsing medication with duration."""
        schedule = self.manager.parse_medication_input("Take Ibuprofen every 6 hours for 3 days")

        assert schedule is not None
        assert schedule.medication_name == "Ibuprofen"
        assert schedule.interval_hours == 6.0
        assert schedule.duration_days == 3

    def test_parse_medication_with_weeks(self):
        """Test parsing medication with week duration."""
        schedule = self.manager.parse_medication_input("Amoxicillin every 8 hours for 2 weeks")

        assert schedule is not None
        assert schedule.medication_name == "Amoxicillin"
        assert schedule.interval_hours == 8.0
        assert schedule.duration_days == 14  # 2 weeks = 14 days

    def test_parse_invalid_input(self):
        """Test that invalid input returns None."""
        schedule = self.manager.parse_medication_input("This is not a valid schedule")
        assert schedule is None

    def test_add_medication_schedule(self):
        """Test adding a medication schedule."""
        result = self.manager.add_medication_schedule("Take Aspirin every 4 hours")

        assert result["success"] is True
        assert result["medication_name"] == "Aspirin"
        assert result["interval_hours"] == 4.0
        assert "next_reminder" in result

    def test_list_active_schedules(self):
        """Test listing active medication schedules."""
        # Add a couple of schedules
        self.manager.add_medication_schedule("Take Medicine A every 6 hours")
        self.manager.add_medication_schedule("Take Medicine B every 12 hours")

        schedules = self.manager.list_active_schedules()

        assert len(schedules) == 2
        assert any(s["medication_name"] == "Medicine A" for s in schedules)
        assert any(s["medication_name"] == "Medicine B" for s in schedules)

    def test_cancel_medication_schedule(self):
        """Test canceling a specific medication schedule."""
        # Add a schedule
        self.manager.add_medication_schedule("Take Test Med every 8 hours")

        # Cancel it
        result = self.manager.cancel_medication_schedule("Test Med")

        assert result["success"] is True
        assert "Test Med" not in self.manager.active_schedules

    def test_cancel_nonexistent_schedule(self):
        """Test canceling a schedule that doesn't exist."""
        result = self.manager.cancel_medication_schedule("Nonexistent")

        assert result["success"] is False
        assert "No active reminder" in result["message"]

    def test_replace_existing_schedule(self):
        """Test that adding a schedule with same medication replaces existing one."""
        # Add initial schedule
        self.manager.add_medication_schedule("Take Tylenol every 4 hours")

        # Add different schedule for same medication
        self.manager.add_medication_schedule("Take Tylenol every 6 hours")

        schedules = self.manager.list_active_schedules()

        # Should only have one Tylenol schedule
        tylenol_schedules = [s for s in schedules if s["medication_name"] == "Tylenol"]
        assert len(tylenol_schedules) == 1
        assert tylenol_schedules[0]["interval_hours"] == 6.0

    def test_reminder_scheduling(self):
        """Test that reminders are actually scheduled (quick test with short interval)."""
        # Use very short interval for testing (0.001 hours ~ 3.6 seconds)
        result = self.manager.add_medication_schedule("Take QuickTest every 0.001 hours")

        assert result["success"] is True

        # Wait a bit to see if timer is set
        time.sleep(0.1)

        # Check that the schedule exists
        assert "Quicktest" in self.manager.active_schedules

        # Clean up
        self.manager.cancel_medication_schedule("Quicktest")
