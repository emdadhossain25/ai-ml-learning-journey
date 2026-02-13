"""
Day 35: Smart File Organizer
Auto-sort files by type into folders
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict


class FileOrganizer:
    """Organize files by type"""
    
    FILE_TYPES = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.ico'],
        'Videos': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'],
        'Documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'],
        'Spreadsheets': ['.xls', '.xlsx', '.csv', '.ods'],
        'Presentations': ['.ppt', '.pptx', '.key', '.odp'],
        'Code': ['.py', '.js', '.java', '.cpp', '.c', '.html', '.css', '.sh', '.rb', '.go', '.rs'],
        'Archives': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'],
        'Executables': ['.exe', '.dmg', '.app', '.deb', '.rpm'],
        'Data': ['.json', '.xml', '.yaml', '.yml', '.sql', '.db'],
    }
    
    def __init__(self, directory='.', dry_run=False):
        self.directory = Path(directory).resolve()
        self.dry_run = dry_run
        self.stats = defaultdict(int)
    
    def get_category(self, file_path):
        """Get category for file"""
        ext = file_path.suffix.lower()
        
        for category, extensions in self.FILE_TYPES.items():
            if ext in extensions:
                return category
        
        return 'Other'
    
    def organize(self):
        """Organize files in directory"""
        
        print(f"\n📂 Organizing: {self.directory}")
        print("="*60)
        
        if self.dry_run:
            print("🔍 DRY RUN MODE - No files will be moved\n")
        
        # Get all files
        files = [f for f in self.directory.iterdir() if f.is_file()]
        
        if not files:
            print("❌ No files found!\n")
            return
        
        print(f"📊 Found {len(files)} files\n")
        
        # Group by category
        by_category = defaultdict(list)
        
        for file in files:
            # Skip hidden files and this script
            if file.name.startswith('.') or file.name == 'organize.py':
                continue
            
            category = self.get_category(file)
            by_category[category].append(file)
        
        # Move files
        for category, category_files in sorted(by_category.items()):
            print(f"\n📁 {category} ({len(category_files)} files):")
            
            # Create category folder
            category_path = self.directory / category
            
            if not self.dry_run and not category_path.exists():
                category_path.mkdir()
            
            # Move each file
            for file in category_files:
                destination = category_path / file.name
                
                # Handle duplicates
                if destination.exists():
                    base = file.stem
                    ext = file.suffix
                    counter = 1
                    
                    while destination.exists():
                        destination = category_path / f"{base}_{counter}{ext}"
                        counter += 1
                
                # Move file
                if self.dry_run:
                    print(f"   • {file.name} → {category}/{destination.name}")
                else:
                    shutil.move(str(file), str(destination))
                    print(f"   ✓ {file.name}")
                
                self.stats[category] += 1
        
        # Summary
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        
        total = sum(self.stats.values())
        
        for category in sorted(self.stats.keys()):
            count = self.stats[category]
            print(f"   {category:15} {count:3} files")
        
        print(f"\n   {'TOTAL':15} {total:3} files")
        
        if self.dry_run:
            print("\n💡 Run without --dry-run to actually move files")
        else:
            print(f"\n✅ Organized {total} files!")
        
        print()
    
    def undo(self):
        """Undo organization (move files back)"""
        
        print(f"\n↩️  Undoing organization in: {self.directory}")
        print("="*60)
        
        moved = 0
        
        # Check each category folder
        for category in self.FILE_TYPES.keys():
            category_path = self.directory / category
            
            if not category_path.exists():
                continue
            
            # Move files back
            for file in category_path.iterdir():
                if file.is_file():
                    destination = self.directory / file.name
                    
                    # Handle duplicates
                    if destination.exists():
                        base = file.stem
                        ext = file.suffix
                        counter = 1
                        
                        while destination.exists():
                            destination = self.directory / f"{base}_restored_{counter}{ext}"
                            counter += 1
                    
                    shutil.move(str(file), str(destination))
                    moved += 1
                    print(f"   ✓ {file.name}")
            
            # Remove empty folder
            if not any(category_path.iterdir()):
                category_path.rmdir()
                print(f"   🗑️  Removed empty folder: {category}")
        
        print(f"\n✅ Restored {moved} files!\n")


def main():
    """CLI"""
    
    import sys
    
    print("\n" + "="*60)
    print("📂 SMART FILE ORGANIZER")
    print("="*60)
    
    # Parse arguments
    if '--help' in sys.argv or '-h' in sys.argv:
        print("""
Usage:
  python3 organize.py [directory] [options]

Options:
  --dry-run    Show what would happen (don't move files)
  --undo       Undo organization (move files back)
  --help       Show this help

Examples:
  python3 organize.py                    # Organize current directory
  python3 organize.py ~/Downloads        # Organize Downloads
  python3 organize.py --dry-run          # Preview changes
  python3 organize.py --undo             # Restore files
""")
        return
    
    # Get directory
    directory = '.'
    for arg in sys.argv[1:]:
        if not arg.startswith('--'):
            directory = arg
            break
    
    # Check flags
    dry_run = '--dry-run' in sys.argv
    undo = '--undo' in sys.argv
    
    # Create organizer
    organizer = FileOrganizer(directory, dry_run)
    
    # Run
    if undo:
        organizer.undo()
    else:
        organizer.organize()


if __name__ == "__main__":
    main()
