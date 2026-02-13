# 📂 Smart File Organizer

**Day 35/100 - Auto-sort messy folders**

## Usage

**Organize current directory:**
```bash
python3 organize.py
```

**Organize specific folder:**
```bash
python3 organize.py ~/Downloads
```

**Preview changes (don't move files):**
```bash
python3 organize.py --dry-run
```

**Undo organization:**
```bash
python3 organize.py --undo
```

## What It Does

Sorts files into folders by type:

- 📷 Images (jpg, png, gif, etc.)
- 🎬 Videos (mp4, avi, mkv, etc.)
- 🎵 Audio (mp3, wav, flac, etc.)
- 📄 Documents (pdf, docx, txt, etc.)
- 📊 Spreadsheets (xlsx, csv, etc.)
- 📈 Presentations (pptx, key, etc.)
- 💻 Code (py, js, java, etc.)
- 📦 Archives (zip, rar, 7z, etc.)
- 🗄️ Data (json, xml, sql, etc.)

## Example

**Before:**
```
Downloads/
├── photo.jpg
├── song.mp3
├── report.pdf
├── video.mp4
└── code.py
```

**After:**
```
Downloads/
├── Images/
│   └── photo.jpg
├── Audio/
│   └── song.mp3
├── Documents/
│   └── report.pdf
├── Videos/
│   └── video.mp4
└── Code/
    └── code.py
```

## Real-World Use

**Clean Downloads folder:**
```bash
python3 organize.py ~/Downloads --dry-run  # Preview
python3 organize.py ~/Downloads             # Do it!
```

**Organize project assets:**
```bash
python3 organize.py ./assets
```

**Clean Desktop:**
```bash
python3 organize.py ~/Desktop --dry-run
```

## Safety

✅ Preview with `--dry-run`  
✅ Undo with `--undo`  
✅ Handles duplicate filenames  
✅ Skips hidden files  

## Built in 15 minutes! ⚡

Use it weekly to stay organized! 🎯
