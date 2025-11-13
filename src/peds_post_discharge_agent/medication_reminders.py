# src/peds_post_discharge_agent/medication_reminders.py
import re
import logging
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class MedicationSchedule:
    """Represents a medication reminder schedule."""
    medication_name: str
    interval_hours: float
    duration_days: Optional[int] = None
    start_time: Optional[datetime] = None
    reminder_count: int = 0
    timer: Optional[threading.Timer] = None


class MedicationReminderManager:
    """Manages medication reminder schedules and alarms."""

    def __init__(self):
        self.active_schedules: dict[str, MedicationSchedule] = {}
        self._lock = threading.Lock()

    def parse_medication_input(self, user_input: str) -> Optional[MedicationSchedule]:
        """
        Parse user input to extract medication schedule information.

        Examples:
        - "Take Zyrtec every 12 hours"
        - "Take Ibuprofen every 6 hours for 3 days"
        - "Amoxicillin every 8 hours for a week"
        """
        # Pattern: medication name + "every X hours" + optional duration
        pattern = r"(?:take\s+)?(\w+(?:\s+\w+)?)\s+every\s+(\d+(?:\.\d+)?)\s*hour[s]?(?:\s+for\s+(\d+)\s+(day|days|week|weeks))?"

        match = re.search(pattern, user_input.lower(), re.IGNORECASE)

        if not match:
            return None

        medication_name = match.group(1).strip().title()
        interval_hours = float(match.group(2))
        duration_spec = match.group(3)
        duration_unit = match.group(4)

        duration_days = None
        if duration_spec and duration_unit:
            duration_days = int(duration_spec)
            # Check if it's weeks
            if 'week' in duration_unit:
                duration_days *= 7

        return MedicationSchedule(
            medication_name=medication_name,
            interval_hours=interval_hours,
            duration_days=duration_days,
            start_time=datetime.now()
        )

    def _trigger_alarm(self, schedule: MedicationSchedule):
        """Trigger an alarm notification for a medication."""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print("\n" + "="*60)
            print("🔔 MEDICATION REMINDER ALARM 🔔")
            print("="*60)
            print(f"Time: {current_time}")
            print(f"Medication: {schedule.medication_name}")
            print(f"Message: It's time to take {schedule.medication_name}")
            print(f"Next reminder in: {schedule.interval_hours} hours")
            print("="*60 + "\n")

            logger.info(f"Medication reminder triggered: {schedule.medication_name}")
            schedule.reminder_count += 1
        except Exception as e:
            logger.error(f"Error triggering alarm: {str(e)}", exc_info=True)

        # Check if we should continue scheduling
        should_continue = True
        if schedule.duration_days:
            elapsed_hours = (datetime.now() - schedule.start_time).total_seconds() / 3600
            total_hours = schedule.duration_days * 24
            if elapsed_hours >= total_hours:
                should_continue = False
                print(f"✓ Medication schedule for {schedule.medication_name} has completed.")
                with self._lock:
                    if schedule.medication_name in self.active_schedules:
                        del self.active_schedules[schedule.medication_name]

        # Schedule next reminder
        if should_continue:
            self._schedule_next_reminder(schedule)

    def _schedule_next_reminder(self, schedule: MedicationSchedule):
        """Schedule the next reminder for a medication."""
        interval_seconds = schedule.interval_hours * 3600

        timer = threading.Timer(
            interval_seconds,
            self._trigger_alarm,
            args=(schedule,)
        )
        timer.daemon = True
        timer.start()

        schedule.timer = timer

    def add_medication_schedule(self, user_input: str) -> dict:
        """
        Add a new medication reminder schedule.

        Returns a dict with status and message.
        """
        schedule = self.parse_medication_input(user_input)

        if not schedule:
            return {
                "success": False,
                "message": "Could not parse medication schedule. Please use format like: 'Take Zyrtec every 12 hours' or 'Take Ibuprofen every 6 hours for 3 days'"
            }

        with self._lock:
            # Cancel existing timer if medication already scheduled
            if schedule.medication_name in self.active_schedules:
                existing = self.active_schedules[schedule.medication_name]
                if existing.timer:
                    existing.timer.cancel()

            # Add new schedule
            self.active_schedules[schedule.medication_name] = schedule

            # Schedule first reminder
            self._schedule_next_reminder(schedule)

            next_reminder_time = datetime.now() + timedelta(hours=schedule.interval_hours)
            duration_info = f" for {schedule.duration_days} days" if schedule.duration_days else ""

            return {
                "success": True,
                "medication_name": schedule.medication_name,
                "interval_hours": schedule.interval_hours,
                "duration_days": schedule.duration_days,
                "next_reminder": next_reminder_time.strftime("%Y-%m-%d %H:%M:%S"),
                "message": f"✓ Reminder set for {schedule.medication_name} every {schedule.interval_hours} hours{duration_info}. First reminder at {next_reminder_time.strftime('%H:%M:%S')}."
            }

    def list_active_schedules(self) -> list[dict]:
        """List all active medication schedules."""
        with self._lock:
            schedules_info = []
            for name, schedule in self.active_schedules.items():
                next_reminder = datetime.now() + timedelta(hours=schedule.interval_hours)
                schedules_info.append({
                    "medication_name": schedule.medication_name,
                    "interval_hours": schedule.interval_hours,
                    "duration_days": schedule.duration_days,
                    "reminders_sent": schedule.reminder_count,
                    "next_reminder": next_reminder.strftime("%Y-%m-%d %H:%M:%S")
                })
            return schedules_info

    def cancel_medication_schedule(self, medication_name: str) -> dict:
        """Cancel a medication reminder schedule."""
        with self._lock:
            medication_name_title = medication_name.title()

            if medication_name_title not in self.active_schedules:
                return {
                    "success": False,
                    "message": f"No active reminder found for {medication_name_title}"
                }

            schedule = self.active_schedules[medication_name_title]
            if schedule.timer:
                schedule.timer.cancel()

            del self.active_schedules[medication_name_title]

            return {
                "success": True,
                "message": f"✓ Reminder for {medication_name_title} has been cancelled."
            }

    def cancel_all_schedules(self):
        """Cancel all active medication schedules."""
        with self._lock:
            for schedule in self.active_schedules.values():
                if schedule.timer:
                    schedule.timer.cancel()
            count = len(self.active_schedules)
            self.active_schedules.clear()
            return {
                "success": True,
                "message": f"✓ Cancelled {count} medication reminder(s)."
            }


# Global singleton instance
_reminder_manager = None

def get_reminder_manager() -> MedicationReminderManager:
    """Get the global medication reminder manager instance."""
    global _reminder_manager
    if _reminder_manager is None:
        _reminder_manager = MedicationReminderManager()
    return _reminder_manager
