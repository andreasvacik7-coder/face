#!/usr/bin/env python3
"""
Test script for Image Quality Manager

Demonstrates how the duplicate detection works on a small sample
"""

from pathlib import Path
from image_quality_manager import ImageQualityManager
import json


def create_test_metadata():
    """Create a small test dataset to demonstrate the functionality"""
    test_metadata = {
        "data/images/test/IMG_123-scaled.jpg": {
            "source_url": "https://example.com/IMG_123-scaled.jpg",
            "website": "test_site",
            "download_date": "2025-09-27 17:00:00",
            "file_size": 2500000  # 2.5MB
        },
        "data/images/test/IMG_123-1024x683.jpg": {
            "source_url": "https://example.com/IMG_123-1024x683.jpg", 
            "website": "test_site",
            "download_date": "2025-09-27 17:00:00",
            "file_size": 150000  # 150KB
        },
        "data/images/test/IMG_123-300x200.jpg": {
            "source_url": "https://example.com/IMG_123-300x200.jpg",
            "website": "test_site", 
            "download_date": "2025-09-27 17:00:00",
            "file_size": 25000  # 25KB
        },
        "data/images/test/IMG_123-150x150.jpg": {
            "source_url": "https://example.com/IMG_123-150x150.jpg",
            "website": "test_site",
            "download_date": "2025-09-27 17:00:00", 
            "file_size": 15000  # 15KB
        },
        "data/images/test/photo-2048x1365.jpg": {
            "source_url": "https://example.com/photo-2048x1365.jpg",
            "website": "test_site",
            "download_date": "2025-09-27 17:00:00",
            "file_size": 800000  # 800KB
        },
        "data/images/test/photo-thumbnail.jpg": {
            "source_url": "https://example.com/photo-thumbnail.jpg", 
            "website": "test_site",
            "download_date": "2025-09-27 17:00:00",
            "file_size": 12000  # 12KB
        }
    }
    return test_metadata


def test_image_quality_manager():
    """Test the Image Quality Manager with sample data"""
    print("🧪 Testing Image Quality Manager")
    print("="*50)
    
    # Create a temporary manager with test data
    manager = ImageQualityManager(
        image_dir=Path("data/images/test"),
        metadata_file=Path("test_metadata.json")
    )
    
    # Load test metadata
    manager.metadata = create_test_metadata()
    
    print("📊 Test Data:")
    for i, (path, data) in enumerate(manager.metadata.items(), 1):
        filename = Path(path).name
        size_mb = data['file_size'] / (1024 * 1024)
        score = manager.calculate_quality_score(filename)
        base_name = manager.extract_base_name(filename)
        print(f"  {i}. {filename}")
        print(f"     Base: {base_name}")
        print(f"     Score: {score}")
        print(f"     Size: {size_mb:.2f}MB")
        print()
    
    print("🔍 Duplicate Analysis:")
    print("-"*30)
    
    # Find duplicates
    duplicates_info = manager.find_all_duplicates()
    
    for base_name, (best_image, duplicates) in duplicates_info.items():
        print(f"\n🎯 Base: {base_name}")
        
        best_filename = Path(best_image).name
        best_score = manager.calculate_quality_score(best_filename)
        best_size = manager.metadata[best_image]['file_size'] / (1024 * 1024)
        
        print(f"   ✅ KEEP: {best_filename}")
        print(f"      Score: {best_score}, Size: {best_size:.2f}MB")
        
        for dup_path in duplicates:
            dup_filename = Path(dup_path).name
            dup_score = manager.calculate_quality_score(dup_filename)
            dup_size = manager.metadata[dup_path]['file_size'] / (1024 * 1024)
            
            print(f"   ❌ REMOVE: {dup_filename}")
            print(f"      Score: {dup_score}, Size: {dup_size:.2f}MB")
    
    print("\n✅ Test completed successfully!")
    
    # Generate report
    print("\n📄 Full Report:")
    print("-"*30)
    print(manager.generate_report())


def demonstrate_integration():
    """Demonstrate how to integrate with existing workflows"""
    print("\n🔧 Integration Examples:")
    print("="*50)
    
    print("""
1. 🎯 Standalone Cleanup:
   python3 image_quality_manager.py --action clean --no-dry-run

2. 📊 Generate Report Only:  
   python3 image_quality_manager.py --action report

3. 🧪 Test Pattern Recognition:
   python3 image_quality_manager.py --action test

4. 🐍 Python Integration:
   
   from image_quality_manager import clean_image_duplicates, generate_duplicate_report
   
   # Generate report
   report = generate_duplicate_report("data/images")
   print(report)
   
   # Clean duplicates (dry run)  
   results = clean_image_duplicates("data/images", dry_run=True)
   print(f"Would remove {results['removal_stats']['files_to_remove']} files")
   
   # Actually clean (remove dry_run=True or set to False)
   # results = clean_image_duplicates("data/images", dry_run=False)

5. 🔄 Integration in image-download.py:
   
   # Add at end of download_and_filter_images() function:
   from image_quality_manager import clean_image_duplicates
   
   # Clean duplicates after download
   print("\\n[*] Cleaning duplicate images...")
   results = clean_image_duplicates(DOWNLOAD_DIR, dry_run=False)
   stats = results['removal_stats']
   
   if stats['files_to_remove'] > 0:
       mb_freed = stats['bytes_to_free'] / (1024 * 1024)
       print(f"[*] Removed {stats['files_to_remove']} duplicate images")
       print(f"[*] Freed {mb_freed:.2f} MB of storage")
   else:
       print("[*] No duplicate images found")
""")


if __name__ == "__main__":
    test_image_quality_manager()
    demonstrate_integration()