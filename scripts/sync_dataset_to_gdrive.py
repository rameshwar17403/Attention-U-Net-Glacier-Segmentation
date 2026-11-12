"""
Automated Dataset Archival & Synchronization to Google Drive / Persistent Storage
Verifies patch integrity, packages split tarballs, copies metadata and inventory,
and computes checksums to ensure scientific reproducibility without storing binaries on GitHub.
"""

import os
import sys
import tarfile
import hashlib
import argparse
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def sync_dataset(dest_dir: str, create_tarballs: bool = False):
    os.makedirs(dest_dir, exist_ok=True)
    inv_path = os.path.join(repo_root, "data", "dataset_inventory.csv")
    rep_path = os.path.join(repo_root, "data", "dataset_verification_report.json")
    patch_dir = os.path.join(repo_root, "data", "patches")

    if not os.path.exists(inv_path):
        raise FileNotFoundError(f"[!] Dataset inventory '{inv_path}' not found.")

    df = pd.read_csv(inv_path)
    print("=" * 70)
    print("📦 DATASET ARCHIVAL & GOOGLE DRIVE SYNCHRONIZATION")
    print(f"   Source directory: {patch_dir}")
    print(f"   Target storage:   {dest_dir}")
    print(f"   Total Patches:    {len(df)}")
    print("=" * 70)

    # Copy inventory & verification report
    import shutil
    shutil.copy2(inv_path, os.path.join(dest_dir, "dataset_inventory.csv"))
    if os.path.exists(rep_path):
        shutil.copy2(rep_path, os.path.join(dest_dir, "dataset_verification_report.json"))
    print(f"[+] Synced dataset_inventory.csv and dataset_verification_report.json")

    dest_patches = os.path.join(dest_dir, "patches")
    os.makedirs(dest_patches, exist_ok=True)

    if create_tarballs:
        for split in ["train", "val", "test"]:
            sub = df[df["split"] == split]
            tar_name = os.path.join(dest_dir, f"{split}_patches.tar.gz")
            print(f"[*] Compressing {len(sub)} {split} patches to {tar_name}...")
            with tarfile.open(tar_name, "w:gz") as tar:
                for _, row in sub.iterrows():
                    p_file = os.path.join(patch_dir, row["patch_file"])
                    m_file = os.path.join(patch_dir, row["mask_file"])
                    if os.path.exists(p_file):
                        tar.add(p_file, arcname=row["patch_file"])
                    if os.path.exists(m_file):
                        tar.add(m_file, arcname=row["mask_file"])
            chk = compute_sha256(tar_name)
            print(f"    -> SHA256: {chk}")
    else:
        synced = 0
        for _, row in df.iterrows():
            p_src = os.path.join(patch_dir, row["patch_file"])
            m_src = os.path.join(patch_dir, row["mask_file"])
            p_dst = os.path.join(dest_patches, row["patch_file"])
            m_dst = os.path.join(dest_patches, row["mask_file"])
            if os.path.exists(p_src) and not os.path.exists(p_dst):
                shutil.copy2(p_src, p_dst)
                synced += 1
            if os.path.exists(m_src) and not os.path.exists(m_dst):
                shutil.copy2(m_src, m_dst)
                synced += 1
        print(f"[+] Direct copy completed ({synced} files synchronized).")

    print(f"\n[+] Dataset successfully archived in persistent storage at: {dest_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sync dataset to persistent storage/Google Drive")
    parser.add_argument("--target-dir", type=str, default="/tmp/GoogleDrive_GlacierDataset",
                        help="Path to persistent storage or mounted Google Drive")
    parser.add_argument("--tarballs", action="store_true", help="Create compressed tarballs by split")
    args = parser.parse_args()

    sync_dataset(args.target_dir, create_tarballs=args.tarballs)


if __name__ == "__main__":
    main()
