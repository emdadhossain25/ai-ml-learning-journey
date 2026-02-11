"""
Day 33: Git Stats Analyzer
Analyze and visualize your git commit history
"""

import subprocess
from collections import Counter, defaultdict
from datetime import datetime
import re


class GitStats:
    """Analyze git repository statistics"""
    
    def get_commits(self):
        """Get all commits"""
        result = subprocess.run(
            ['git', 'log', '--pretty=format:%h|%an|%ae|%ad|%s', '--date=short'],
            capture_output=True,
            text=True
        )
        
        commits = []
        for line in result.stdout.split('\n'):
            if line:
                hash, author, email, date, message = line.split('|', 4)
                commits.append({
                    'hash': hash,
                    'author': author,
                    'email': email,
                    'date': date,
                    'message': message
                })
        
        return commits
    
    def analyze(self):
        """Analyze commit history"""
        commits = self.get_commits()
        
        if not commits:
            return None
        
        # Stats
        total = len(commits)
        authors = Counter(c['author'] for c in commits)
        by_date = Counter(c['date'] for c in commits)
        by_day = Counter(datetime.strptime(c['date'], '%Y-%m-%d').strftime('%A') for c in commits)
        
        # Commit types (conventional commits)
        types = defaultdict(int)
        for c in commits:
            msg = c['message'].lower()
            if msg.startswith('feat'): types['Features'] += 1
            elif msg.startswith('fix'): types['Fixes'] += 1
            elif msg.startswith('docs'): types['Docs'] += 1
            elif msg.startswith('refactor'): types['Refactors'] += 1
            elif msg.startswith('test'): types['Tests'] += 1
            elif msg.startswith('chore'): types['Chores'] += 1
            else: types['Other'] += 1
        
        # Streaks
        dates = sorted(set(c['date'] for c in commits))
        current_streak = 1
        max_streak = 1
        
        for i in range(1, len(dates)):
            prev = datetime.strptime(dates[i-1], '%Y-%m-%d')
            curr = datetime.strptime(dates[i], '%Y-%m-%d')
            
            if (curr - prev).days == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        
        return {
            'total': total,
            'authors': authors,
            'by_date': by_date,
            'by_day': by_day,
            'types': types,
            'max_streak': max_streak,
            'first_commit': commits[-1]['date'],
            'last_commit': commits[0]['date'],
            'recent': commits[:5]
        }


def print_stats(stats):
    """Pretty print statistics"""
    
    print("\n" + "="*60)
    print("📊 GIT STATISTICS")
    print("="*60)
    
    print(f"\n📈 OVERVIEW:")
    print(f"   Total commits: {stats['total']}")
    print(f"   First commit: {stats['first_commit']}")
    print(f"   Last commit: {stats['last_commit']}")
    print(f"   Longest streak: {stats['max_streak']} days 🔥")
    
    print(f"\n👤 CONTRIBUTORS:")
    for author, count in stats['authors'].most_common(5):
        bar = "█" * min(count // 2, 40)
        print(f"   {author:20} {bar} {count}")
    
    print(f"\n📅 COMMITS BY DAY:")
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in days_order:
        count = stats['by_day'].get(day, 0)
        bar = "█" * min(count // 2, 30)
        print(f"   {day:10} {bar} {count}")
    
    print(f"\n🏷️  COMMIT TYPES:")
    for type_name, count in sorted(stats['types'].items(), key=lambda x: -x[1]):
        bar = "█" * min(count // 2, 30)
        print(f"   {type_name:12} {bar} {count}")
    
    print(f"\n📝 RECENT COMMITS:")
    for c in stats['recent']:
        print(f"   {c['hash']} {c['date']} {c['message'][:50]}")
    
    print("\n" + "="*60)


def main():
    """CLI"""
    
    print("\n🔍 Analyzing git repository...")
    
    try:
        analyzer = GitStats()
        stats = analyzer.analyze()
        
        if stats:
            print_stats(stats)
            
            # Save report
            with open('git_report.txt', 'w') as f:
                f.write(f"Git Statistics Report\n")
                f.write(f"Generated: {datetime.now()}\n\n")
                f.write(f"Total commits: {stats['total']}\n")
                f.write(f"Contributors: {len(stats['authors'])}\n")
                f.write(f"Longest streak: {stats['max_streak']} days\n")
            
            print("\n💾 Report saved to: git_report.txt\n")
        else:
            print("❌ No commits found")
    
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
