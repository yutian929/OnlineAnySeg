import os
import numpy as np
from tqdm import tqdm
import shutil

"""Extract GT poses for selected SceneNN sequences"""

intrinsic_file = "./preprocess/SceneNN/intrinsic/intrinsic_depth.txt"  # intrinsic file
dataset_dir = "./data/SceneNN"


def read_file_in_chunks(filename):
    chunks = []  # 存储所有数据块
    with open(filename, 'r') as file:
        chunk = []  # 临时存储每个数据块的列表
        for i, line in enumerate(file):
            chunk.append(line.strip())  # 去除行尾换行符并添加到当前数据块
            if (i + 1) % 5 == 0:  # 每5行分成一个块
                chunks.append(chunk)
                chunk = []  # 重置为下一个数据块
        if chunk:  # 如果最后的块不足5行，也加入chunks
            chunks.append(chunk)
    return chunks


def copy_intrinsic_file(src_file, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    shutil.copy(src_file, dst_dir)


def main(scene_id):
    seq_dir = os.path.join(dataset_dir, scene_id)
    if not os.path.exists(seq_dir):
        print("%s does not exist!" % seq_dir)
        return

    # Step 1: copy intrinsic file to the target directory of given scene
    dst_intrinsic_dir = os.path.join(seq_dir, "intrinsic")
    copy_intrinsic_file(intrinsic_file, dst_intrinsic_dir)

    # Step 2: load raw trajectory file into blocks
    traj_file = os.path.join(seq_dir, "trajectory.log")  # trajectory file
    chunks = read_file_in_chunks(traj_file)

    # Step 3: for each block, extract its frame_ID and pose_c2w
    frame_poses = {}
    for i, chunk in enumerate(chunks):
        head_line = chunk[0].strip()
        frame_id = int( head_line.split()[-1] )

        pose_rows = []
        for j in range(1, 5):
            element_list = chunk[j].strip().split()
            element_list = list(map(float, element_list))  # str --> float
            pose_row = np.asarray(element_list)
            pose_rows.append(pose_row)
        pose_c2w = np.stack(pose_rows, axis=0)  # ndarray(4, 4)

        frame_poses[frame_id] = pose_c2w

    # Step 4: for each frame with pose_c2w, write it into target directory
    output_dir = os.path.join(seq_dir, "pose")
    os.makedirs(output_dir, exist_ok=True)

    frame_num = len(frame_poses)
    for frame_id, pose_c2w in tqdm(frame_poses.items()):
        output_path = os.path.join(output_dir, "%d.txt" % frame_id)
        np.savetxt(output_path, pose_c2w, fmt='%.8f')

    print("Finished write poses of %d frames into: %s" % (frame_num, output_dir))


if __name__ == '__main__':
    scene_id_list = ["005", "011", "015", "030", "054", "080", "089", "093", "096", "243", "263", "322"]
    scene_num = len(scene_id_list)

    for i, scene_id in enumerate(scene_id_list):
        print("\n########################### Begin to process sequence: %s (%d / %d)..." % (scene_id, i+1, scene_num))

        main(scene_id)

        print("########################### Finished processing sequence: %s (%d / %d)!!!" % (scene_id, i+1, scene_num))

