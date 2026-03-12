import pandas as pd
import os

def prepare_chatbot_data():
    # File paths
    report_path = r"C:\xray\data\train_metadata.csv"
    img_metadata_path = r"C:\xray\New folder\train_metadata_new.csv"
    output_path = r"C:\xray\model\final_chatbot_train.csv"

    print("Reading files...")
    df_reports = pd.read_csv(report_path)
    df_imgs = pd.read_csv(img_metadata_path)

    print("Merging on StudyInstanceUid...")
    combined_df = pd.merge(df_reports, df_imgs, on='StudyInstanceUid', how='inner')

    # Long Path Fix Logic: Adding \\?\ for Windows Long Paths
    def fix_long_path(p):
        if pd.isna(p): return p
        # Absolute path banavi ne prefix lagadvo
        abs_path = os.path.abspath(p)
        if not abs_path.startswith("\\\\?\\"):
            return "\\\\?\\" + abs_path
        return abs_path

    print("Fixing long paths with \\\\?\\ prefix...")
    # 'img_path' column mathi path sudharvo
    combined_df['image_path'] = combined_df['img_path'].apply(fix_long_path)

    # Findings handle karva
    findings_col = 'Findings_x' if 'Findings_x' in combined_df.columns else 'Findings'
    combined_df['chatbot_caption'] = combined_df[findings_col].fillna("The chest x-ray appears normal.")

    # Final selection
    final_df = combined_df[['StudyInstanceUid', 'image_path', 'chatbot_caption']]
    
    final_df.to_csv(output_path, index=False)
    print(f"SUCCESS! Created {output_path}")
    print(f"Sample Path: {final_df['image_path'].iloc[0]}")

if __name__ == "__main__":
    prepare_chatbot_data()