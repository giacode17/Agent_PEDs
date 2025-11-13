#!/usr/bin/env python3
"""
Test script for medication reminder functionality.
Demonstrates setting reminders and waiting for alarms to trigger.
"""
import time
from src.peds_post_discharge_agent.tools import (
    set_medication_reminder,
    list_medication_reminders,
    cancel_medication_reminder
)


def main():
    print("=" * 60)
    print("MEDICATION REMINDER SYSTEM TEST")
    print("=" * 60)
    print()

    # Test 1: Set a reminder for Zyrtec
    print("Test 1: Setting reminder for Zyrtec every 12 hours")
    result = set_medication_reminder("Take Zyrtec every 12 hours")
    print(result)
    print()

    time.sleep(2)

    # Test 2: Set a reminder for Ibuprofen with duration
    print("Test 2: Setting reminder for Ibuprofen every 6 hours for 3 days")
    result = set_medication_reminder("Take Ibuprofen every 6 hours for 3 days")
    print(result)
    print()

    time.sleep(2)

    # Test 3: List active reminders
    print("Test 3: Listing all active reminders")
    result = list_medication_reminders()
    print(result)
    print()

    time.sleep(2)

    # Test 4: Test with short interval for demo (10 seconds)
    print("Test 4: Setting a quick reminder for demo (Amoxicillin every 0.0028 hours = ~10 seconds)")
    result = set_medication_reminder("Take Amoxicillin every 0.0028 hours")
    print(result)
    print()

    # Wait for the alarm to trigger
    print("Waiting 12 seconds for the Amoxicillin alarm to trigger...")
    print("(You should see an alarm notification appear in the console)")
    print()
    time.sleep(12)

    # Test 5: Cancel a reminder
    print("\nTest 5: Cancelling Amoxicillin reminder")
    result = cancel_medication_reminder("Amoxicillin")
    print(result)
    print()

    # Test 6: List reminders after cancellation
    print("Test 6: Listing active reminders after cancellation")
    result = list_medication_reminders()
    print(result)
    print()

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nNote: The Zyrtec and Ibuprofen reminders are still active.")
    print("They will trigger alarms in 12 hours and 6 hours respectively.")
    print("You can cancel them by calling cancel_medication_reminder()")
    print()


if __name__ == "__main__":
    main()
