import csv
import os

def process_files():
    configs = [
        {
            'input': "stage2_rewrites(benign).txt",
            'output': "top_30_stage2_rewrites(benign).csv",
            'reverse': True, # Descending (highest first)
            'label': 'benign'
        },
        {
            'input': "stage2_rewrites(jailbreak).txt",
            'output': "top_30_stage2_rewrites(jailbreak).csv",
            'reverse': False, # Ascending (lowest first)
            'label': 'jailbreak'
        }
    ]

    for config in configs:
        file_path = config['input']
        if not os.path.exists(file_path):
            print(f"Skipping {config['label']}: File not found - {file_path}")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if 'score' not in reader.fieldnames:
                    print(f"Skipping {config['label']}: 'score' column not found in {file_path}")
                    continue
                
                rows = list(reader)
            
            # Filter and convert scores
            valid_rows = []
            for row in rows:
                try:
                    row['score'] = float(row['score'])
                    valid_rows.append(row)
                except ValueError:
                    continue
            
            # Sort
            valid_rows.sort(key=lambda x: x['score'], reverse=config['reverse'])
            
            # Take top 30
            top_30 = valid_rows[:30]
            
            # Save
            output_path = config['output']
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(top_30)
            
            print(f"[{config['label']}] Extracted top 30 to {output_path}")
            
            # Preview
            print(f"Preview for {config['label']} (Top 3):")
            for i, row in enumerate(top_30[:3]):
                text_snippet = row['text'][:50] + "..." if len(row['text']) > 50 else row['text']
                print(f"  {i+1}. Score: {row['score']:.4f} | Text: {text_snippet}")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing {config['label']}: {e}")

if __name__ == "__main__":
    process_files()
