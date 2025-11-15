#!/usr/bin/env python3
"""
Demo script to show how to check medication schedules.
This demonstrates the list_active_schedules() functionality.
"""
from peds_post_discharge_agent.medication_reminders import get_reminder_manager
import json

def main():
    # Get the global reminder manager
    manager = get_reminder_manager()

    print("="*70)
    print("  Medication Schedule Checker Demo")
    print("="*70)
    print()

    # Example: Add some medication schedules
    print("Adding medication schedules...")
    print("-" * 70)

    schedules_to_add = [
        "Take Ibuprofen every 6 hours for 3 days",
        "Take Amoxicillin every 8 hours for 2 weeks",
        "Take Zyrtec every 12 hours"
    ]

    for schedule_input in schedules_to_add:
        result = manager.add_medication_schedule(schedule_input)
        if result["success"]:
            print(f"✓ {result['message']}")
        else:
            print(f"✗ Failed: {result['message']}")

    print()
    print("="*70)
    print("  ACTIVE MEDICATION SCHEDULES")
    print("="*70)
    print()

    # Check active schedules
    active_schedules = manager.list_active_schedules()

    if not active_schedules:
        print("No active medication schedules.")
    else:
        print(f"Found {len(active_schedules)} active schedule(s):\n")

        for i, schedule in enumerate(active_schedules, 1):
            print(f"{i}. {schedule['medication_name']}")
            print(f"   Interval: Every {schedule['interval_hours']} hours")
            if schedule['duration_days']:
                print(f"   Duration: {schedule['duration_days']} days")
            else:
                print(f"   Duration: Ongoing (no end date)")
            print(f"   Reminders sent: {schedule['reminders_sent']}")
            print(f"   Next reminder: {schedule['next_reminder']}")
            print()

    print("="*70)
    print("  RAW DATA (JSON format)")
    print("="*70)
    print(json.dumps(active_schedules, indent=2))
    print()

    # Example: Cancel a specific medication
    print("="*70)
    print("  CANCELING A SCHEDULE")
    print("="*70)
    cancel_result = manager.cancel_medication_schedule("Zyrtec")
    print(f"{cancel_result['message']}\n")

    # Check active schedules again
    print("Remaining active schedules:")
    remaining = manager.list_active_schedules()
    print(f"Count: {len(remaining)}")
    for schedule in remaining:
        print(f"  - {schedule['medication_name']}")

    print()
    print("="*70)
    print("  CLEANUP")
    print("="*70)
    cleanup_result = manager.cancel_all_schedules()
    print(f"{cleanup_result['message']}")
    print()

if __name__ == "__main__":
    main()
