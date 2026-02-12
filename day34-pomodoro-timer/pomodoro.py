"""
Day 34: Terminal Pomodoro Timer
Focus timer with notifications
"""

import time
import os
import sys
from datetime import datetime, timedelta


class PomodoroTimer:
    """Simple Pomodoro timer"""
    
    def __init__(self):
        self.work_duration = 25 * 60  # 25 minutes
        self.break_duration = 5 * 60  # 5 minutes
        self.long_break = 15 * 60     # 15 minutes
        self.sessions_until_long_break = 4
    
    def clear_screen(self):
        """Clear terminal"""
        os.system('clear' if os.name != 'nt' else 'cls')
    
    def notify(self, message):
        """Send notification"""
        # macOS notification
        os.system(f'osascript -e \'display notification "{message}" with title "Pomodoro Timer"\'')
        # Also print
        print(f"\n🔔 {message}\n")
    
    def format_time(self, seconds):
        """Format seconds as MM:SS"""
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:02d}:{secs:02d}"
    
    def countdown(self, duration, label):
        """Run countdown timer"""
        end_time = datetime.now() + timedelta(seconds=duration)
        
        while True:
            remaining = (end_time - datetime.now()).total_seconds()
            
            if remaining <= 0:
                break
            
            # Draw timer
            self.clear_screen()
            print("="*50)
            print(f"⏱️  POMODORO TIMER")
            print("="*50)
            print(f"\n{label}")
            print(f"\n   {self.format_time(int(remaining))}")
            
            # Progress bar
            progress = 1 - (remaining / duration)
            bar_length = 40
            filled = int(bar_length * progress)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\n   [{bar}] {int(progress * 100)}%")
            
            print("\n" + "="*50)
            print("Press Ctrl+C to stop")
            
            time.sleep(1)
    
    def run(self):
        """Run Pomodoro sessions"""
        
        session = 1
        
        print("\n" + "="*50)
        print("🍅 POMODORO TIMER")
        print("="*50)
        print("\nFocus: 25 min | Break: 5 min | Long break: 15 min")
        print("\nPress Enter to start...")
        input()
        
        try:
            while True:
                # Work session
                self.notify(f"Session {session} - Time to FOCUS! 💪")
                self.countdown(self.work_duration, f"🎯 FOCUS TIME - Session {session}")
                self.notify("Great work! Time for a break! ☕")
                
                # Break
                if session % self.sessions_until_long_break == 0:
                    self.countdown(self.long_break, "☕ LONG BREAK - You earned it!")
                else:
                    self.countdown(self.break_duration, "☕ SHORT BREAK - Relax!")
                
                self.notify(f"Break over! Ready for session {session + 1}? 🚀")
                
                session += 1
                
                # Pause between sessions
                print("\nPress Enter for next session (or Ctrl+C to stop)...")
                input()
        
        except KeyboardInterrupt:
            print(f"\n\n✅ Completed {session} session(s)!")
            print(f"⏱️  Focus time: {session * 25} minutes\n")


def quick_timer(minutes):
    """Quick custom timer"""
    print(f"\n⏱️  {minutes} minute timer starting...\n")
    
    duration = minutes * 60
    end_time = datetime.now() + timedelta(seconds=duration)
    
    try:
        while True:
            remaining = (end_time - datetime.now()).total_seconds()
            
            if remaining <= 0:
                break
            
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            
            print(f"\r   {mins:02d}:{secs:02d}", end='', flush=True)
            time.sleep(1)
        
        print("\n\n🔔 Time's up!\n")
        os.system('osascript -e \'display notification "Timer finished!" with title "Timer"\'')
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Timer stopped\n")


def main():
    """CLI"""
    
    if len(sys.argv) > 1:
        # Quick timer mode
        try:
            minutes = int(sys.argv[1])
            quick_timer(minutes)
        except:
            print("Usage: python3 pomodoro.py <minutes>")
    else:
        # Pomodoro mode
        timer = PomodoroTimer()
        timer.run()


if __name__ == "__main__":
    main()
