"""
Interactive CLI Interface for College Enquiry Chatbot.
Run this script to chat with the bot directly in your terminal.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chatbot import CollegeChatbotEngine


def print_banner(bot: CollegeChatbotEngine):
    print("=" * 72)
    print(f"      🎓 WELCOME TO {bot.college_name.upper()} ENQUIRY BOT 🎓")
    print("=" * 72)
    print("Ask any question regarding:")
    print("  • 📚 Courses      (UG, PG, eligibility, curriculum, duration)")
    print("  • 💰 Fees         (Tuition, hostel, exams, scholarships, payment modes)")
    print("  • 📝 Exams        (Datesheets, hall tickets, grading, revaluation, backlogs)")
    print("  • 🏛️  Departments  (CSE, ECE, Mech, Civil, Management, lab setups)")
    print("  • ⏰ Timings      (College hours, library, office, gym & sports)")
    print("  • 🏢 Facilities   (Hostel, library, Wi-Fi, transport, food court, sports)")
    print("-" * 72)
    print("Commands:")
    print("  /domains   - List supported enquiry domains")
    print("  /clear     - Reset conversation history")
    print("  /help      - Show available commands")
    print("  /exit      - Quit the chat session")
    print("=" * 72)
    print()


def run_cli():
    bot = CollegeChatbotEngine()
    print_banner(bot)

    # Initial greeting
    init_res = bot.process_query("hello")
    print(f"🤖 Bot: {init_res['response']}\n")
    if init_res.get("suggestions"):
        chips = " | ".join(f"[{s}]" for s in init_res["suggestions"])
        print(f"   💡 Suggested topics: {chips}\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["/exit", "exit", "quit", ":q"]:
                bye_res = bot.process_query("bye")
                print(f"\n🤖 Bot: {bye_res['response']}\n")
                break

            elif user_input.lower() == "/clear":
                bot.reset_conversation()
                print("\n🧹 Conversation history cleared.\n")
                continue

            elif user_input.lower() == "/domains":
                print("\n🎓 Supported Domains:")
                for cat in bot.categories:
                    print(f"   - {cat.capitalize()}")
                print()
                continue

            elif user_input.lower() == "/help":
                print_banner(bot)
                continue

            # Process query
            res = bot.process_query(user_input)

            print(f"\n🤖 Bot: {res['response']}")
            print(f"   [Category: {res['category']} | Intent: {res['intent']} | Confidence: {res['confidence'] * 100:.1f}%]")

            if res.get("entities"):
                ents_str = ", ".join(f"{k}: {v}" for k, v in res["entities"].items())
                print(f"   [Extracted Slots: {ents_str}]")

            if res.get("suggestions"):
                chips = " | ".join(f"[{s}]" for s in res["suggestions"])
                print(f"   💡 Next Suggestions: {chips}")

            print()

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended. Goodbye!")
            break


if __name__ == "__main__":
    run_cli()
