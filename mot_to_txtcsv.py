import pandas as pd
import os

# --- Configuration ---
MOT_DIR = '.' 
OUTPUT_DIR = 'txt' 
DESIRED_COLUMNS = [
    'time',
    'hip_flexion_r', 
    'knee_angle_r', 
    'hip_flexion_l', 
    'knee_angle_l'
]
SKIP_ROWS = 10 # Skips the MOT file header (typically 10 lines)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all .mot files
mot_files = [f for f in os.listdir(MOT_DIR) if f.endswith('.mot')]

if not mot_files:
    print("❌ No .mot files found to process.")
else:
    print(f"✅ Found {len(mot_files)} .mot file(s). Starting conversion...")
    
    # Process each file
    for mot_file in mot_files:
        output_filename = os.path.splitext(mot_file)[0] + '.txt'
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        try:
            # 1. Read the MOT file (uses r'\s+' for variable whitespace separation)
            df = pd.read_csv(
                mot_file, 
                sep=r'\s+', 
                skiprows=SKIP_ROWS
            )
            
            # 2. Select only the desired columns
            df_filtered = df[DESIRED_COLUMNS]

            # 3. Save to output file (CSV format: comma separator, decimal point)
            df_filtered.to_csv(
                output_path, 
                index=False, 
                sep=',', 
                decimal='.'
            )
            
            print(f"   -> Converted: **{mot_file}** to **{output_path}**")

        except KeyError as e:
            print(f"   ❌ Error processing {mot_file}: Column {e} not found.")
        except Exception as e:
            print(f"   ❌ General error processing {mot_file}: {e}")
            
    print("\n🎉 Batch processing complete.")