#!/usr/bin/env python3
"""
Helper script that runs the "Create Final Transcript" workflow for
"""

from dotenv import load_dotenv

from run_final_transcript import LANGUAGE, process_conversation


def main():
    load_dotenv()

    targets = [92]
    succeeded = []
    failed = []

    for num in targets:
        try:
            if process_conversation(num, LANGUAGE):
                succeeded.append(num)
            else:
                failed.append(num)
        except Exception as exc:  # pylint: disable=broad-except
            failed.append(num)
            print(f"❌ Conversation_{num} failed: {exc}")

    print("==============================================")
    print("Batch Summary")
    print(f"✅ Successful: {succeeded}" if succeeded else "✅ Successful: None")
    print(f"❌ Failed: {failed}" if failed else "❌ Failed: None")


if __name__ == "__main__":
    main()

