import os
import matplotlib.pyplot as plt
from fairmotion.data import bvh


def load_skl():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dataPath = os.path.join(root_dir, 'single-person', 'data', 'raw_data', 'indiv_data')
    print(f'dataPath: {dataPath}')

    sub_names = ['X05', 'X07', 'X08', 'X09']
    motion_set = {}

    for sub_name in sub_names:
        seqs_path = os.path.join(dataPath, sub_name)
        print(f'Loading... {seqs_path}')
        seqs_list = os.listdir(seqs_path)
        motion_list = []
        for seq in seqs_list:
            seq_bvh_path = os.path.join(seqs_path, seq)
            motion = bvh.load(seq_bvh_path)
            positions = motion.positions(local=False)
            positions_scale = positions / 1000
            motion_list.append(positions_scale)
        motion_set[sub_name] = motion_list
    
    return motion_set


def visualize_frame(frame):
    # skeleton = motion.skeleton  # 親子関係を取得
    # --- 3D表示 ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ノードを点で表示
    ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], c='b', s=20)
    # 各ノードにindexを表示
    for i, (x, y, z) in enumerate(frame):
        ax.text(x, y, z, str(i), color='black', fontsize=8, ha='center', va='bottom')

    # 各ジョイントを線でつなぐ
    # for joint in skeleton.joints:
    #     if joint.parent is not None:
    #         p1 = frame[joint.index]
    #         p2 = frame[joint.parent.index]
    #         ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{sub_name} - frame 0")

    # 軸スケールをそろえる（歪み防止）
    max_range = (frame.max(axis=0) - frame.min(axis=0)).max() / 2.0
    mid = frame.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    # ✅ 視点を調整（elev = 上下角度, azim = 水平角度）
    elev= 18
    azim = -10
    ax.view_init(elev=elev, azim=azim)  # ←ここを変えると視点が変わる！

    plt.savefig(f'./elev-{elev}-azim-{azim}.png', dpi=300)
    plt.show()

if __name__=='__main__':
    motion_set = load_skl()

    # --- 例として X05 の最初のサンプルを取り出す ---
    sub_name = 'X05'
    sample_idx = 0
    motion = motion_set[sub_name][sample_idx]  # motion.shape は (frames, joints, 3)
    frame = motion[0]  # 0フレーム目の (joints, 3)
    visualize_frame(frame)