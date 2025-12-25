import os

dataset_dir = "dataset"
images_dir = os.path.join(dataset_dir, "images")

print("=" * 60)
print("📁 CHECKING IMAGES DIRECTORY")
print("=" * 60)

if os.path.exists(images_dir):
    all_files = os.listdir(images_dir)
    print(f"\nTotal files in {images_dir}: {len(all_files)}")
    
    if len(all_files) == 0:
        print("❌ Directory is EMPTY!")
        print("\n💡 You need to add image files to this directory.")
        print("   Expected format: potato_001_top.jpg, potato_001_left.jpg, etc.")
    else:
        print("\nAll files found:")
        for f in sorted(all_files):
            file_path = os.path.join(images_dir, f)
            size = os.path.getsize(file_path)
            print(f"   - {f} ({size} bytes)")
        
        # Check extensions
        extensions = {}
        for f in all_files:
            ext = os.path.splitext(f)[1].lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        
        print(f"\nFile extensions found:")
        for ext, count in sorted(extensions.items()):
            print(f"   {ext or '(no extension)'}: {count} files")
        
        if '.jpg' not in extensions and '.jpeg' not in extensions:
            print("\n⚠️  No .jpg or .jpeg files found!")
            print("   If you have images with different extensions (png, JPG, JPEG),")
            print("   you can either:")
            print("   1. Rename them to .jpg")
            print("   2. Update the code to accept other extensions")
else:
    print(f"❌ Directory does not exist: {images_dir}")
    print("   Please create it and add your potato images.")

print("\n" + "=" * 60)