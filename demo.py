import random
import re
import textwrap
import time
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import igraph as ig



def IC(G, S, p=0.5, mc=1000):
    """
    Mô phỏng quá trình lan truyền ảnh hưởng trong mạng xã hội theo mô hình Independent Cascade.
    G: Đồ thị mạng xã hội
    S: Tập hạt giống
    p: Xác suất lan truyền
    mc: Số lần mô phỏng Monte Carlo
    """
    spread = []
    for _ in range(mc):
        new_active, A = S[:], S[:]
        while new_active:
            temp = G.loc[G['source'].isin(new_active)]
            targets = temp['target'].tolist()
            success = np.random.uniform(0, 1, len(targets)) < p
            new_ones = np.extract(success, targets)
            new_active = list(set(new_ones) - set(A))
            A += new_active
        spread.append(len(A))
    return np.mean(spread)


def get_RRS(G, p):
    """
    Xây dựng tập ngẫu nhiên các nút có thể bị ảnh hưởng ngược từ một nút nguồn.
    G: Đồ thị mạng xã hội
    p: Xác suất lan truyền
    """
    source = random.choice(np.unique(G['source']))
    g = G.copy().loc[np.random.uniform(0, 1, G.shape[0]) < p]
    new_nodes, RRS0 = [source], [source]
    while new_nodes:
        temp = g.loc[g['target'].isin(new_nodes)]
        temp = temp['source'].tolist()
        RRS = list(set(RRS0 + temp))
        new_nodes = list(set(RRS) - set(RRS0))
        RRS0 = RRS[:]
    return RRS


def find_substring_in_parentheses(s):
    # Biểu thức chính quy tìm chuỗi giữa dấu '(' và ')'
    pattern = r'\((.*?)\)'

    # Tìm tất cả các chuỗi con trong dấu '(' và ')'
    matches = re.findall(pattern, s)

    return matches

def celf(G, k, p=0.5, mc=1000):
    """
    Áp dụng thuật toán CELF để tìm tập hạt giống tối ưu.
    G: Đồ thị mạng xã hội
    k: Số lượng nút cần chọn
    p: Xác suất lan truyền
    mc: Số lần mô phỏng Monte Carlo
    """
    start_time = time.time()
    candidates = np.unique(G['source'])
    marg_gain = [IC(G, [node], p=p, mc=mc) for node in candidates]
    Q = sorted(zip(candidates, marg_gain), key=lambda x: x[1], reverse=True)
    S, spread = [Q[0][0]], Q[0][1]
    Q = Q[1:]
    timelapse = []

    for _ in range(k - 1):
        check = False
        while not check:
            current = Q[0][0]
            Q[0] = (current, IC(G, S + [current], p=p, mc=mc) - spread)
            Q = sorted(Q, key=lambda x: x[1], reverse=True)
            check = Q[0][0] == current
        S.append(Q[0][0])
        spread = Q[0][1]
        Q = Q[1:]
        timelapse.append(time.time() - start_time)

    total_time = time.time() - start_time
    return sorted(S), timelapse, total_time

def ris(G, k, p=0.5, mc=10000):
    """
    Đầu vào:
        G: DataFrame chứa các cạnh có hướng của đồ thị. Các cột bao gồm ['source', 'target'].
        k: Số lượng nút cần chọn trong seed set.
        p: Xác suất lan truyền của ảnh hưởng (disease propagation probability).
        mc: Số lần tạo Reverse Reachable Sets (RRSs) (mặc định là 1000).
    Đầu ra:
        SEED: Tập seed set gồm các nút tối ưu giải quyết bài toán Influence Maximization.
        timelapse: Danh sách thời gian ghi lại trong quá trình thực hiện.
        total_time: Tổng thời gian thực hiện thuật toán.
    """
    start_time = time.time()  # Ghi nhận thời gian bắt đầu quá trình
    R = [get_RRS(G, p) for _ in range(mc)]  # Tạo mc RRS ngẫu nhiên
    SEED, timelapse = [], []  # Khởi tạo danh sách chứa các nút seed và thời gian thực hiện

    for _ in range(k):
        # Tạo một danh sách phẳng của tất cả các nút trong các RRS đã tạo
        flat_list = [item for sublist in R for item in sublist]
        # Tìm nút xuất hiện nhiều nhất trong danh sách, đây là nút có ảnh hưởng lớn nhất
        seed = Counter(flat_list).most_common()[0][0]
        SEED.append(seed)  # Thêm nút vào seed set
        # Loại bỏ các RRS chứa nút đã chọn từ tập hợp R
        R = [rrs for rrs in R if seed not in rrs]
        timelapse.append(time.time() - start_time)  # Lưu thời gian tính toán từ lúc bắt đầu

    total_time = time.time() - start_time  # Tính tổng thời gian thực hiện
    return sorted(SEED), timelapse, total_time

def validate_graph_params(num_nodes, num_edges):
    """
    Kiểm tra điều kiện của số nút và số cạnh trong đồ thị mạng xã hội.

    Parameters:
        num_nodes (int): Số lượng nút.
        num_edges (int): Số lượng cạnh từ mỗi nút mới.

    Returns:
        bool: True nếu điều kiện hợp lệ, ngược lại False.
        str: Thông báo lỗi nếu có.
    """
    if num_nodes <= 0:
        return False, "**Số lượng nút phải lớn hơn 0.**"
    if num_edges <= 0:
        return False, "**Số lượng cạnh phải lớn hơn 0.**"
    if num_edges >= num_nodes:
        return False, "**Số lượng cạnh phải nhỏ hơn số lượng nút.**"
    return True, "**Các thông số hợp lệ.**"


def create_G(n=100, m=3):
    # Tạo đồ thị Barabási-Albert
    G = ig.Graph.Barabasi(n=n, m=m, directed=True)

    # Tạo DataFrame chứa thông tin cạnh
    source_nodes = [edge.source for edge in G.es]
    target_nodes = [edge.target for edge in G.es]
    df = pd.DataFrame({'source': source_nodes, 'target': target_nodes})
    return G, df

def show_G(G):
    # Gán nhãn cho các nút
    G.vs["label"] = [str(i) for i in range(G.vcount())]

    # Vẽ đồ thị sử dụng matplotlib
    visual_style = {
        "vertex_size": 25,
        "vertex_label": G.vs["label"],
        "vertex_label_size": 12,
        "edge_color": "#B3CDE3",
        "vertex_color": "#FBB4AE",
        "layout": G.layout("kk"),  # Kamada-Kawai layout
    }

    fig, ax = plt.subplots(figsize=(8, 8))
    ig.plot(
        G,
        target=ax,
        bbox=(0, 0, 600, 600),
        **visual_style
    )
    plt.title("Đồ thị MXH G", fontsize=16)
    plt.show()


def print_centered_text():
    text = (
        "CHƯƠNG TRÌNH THỰC NGHIỆM MÔ PHỎNG BÀI TOÁN TỐI ĐA HÓA ẢNH HƯỞNG MẠNG XÃ HỘI "
        "SỬ DỤNG THUẬT TOÁN CEFL VÀ RIS"
    )

    # Sử dụng textwrap để chia dòng (line width là 80 ký tự)
    wrapped_text = textwrap.fill(text, width=80)

    # Tạo mỗi dòng căn giữa
    lines = wrapped_text.split('\n')
    terminal_width = 80  # Chiều rộng giả định của màn hình console
    for line in lines:
        print(line.center(terminal_width))


def validate_inputs(k, p=0.5, mc=1000):
    """
    Kiểm tra các tham số đầu vào của thuật toán CELF.

    Parameters:
        k: int
            Số lượng nút cần chọn, phải là số nguyên dương.
        p: float
            Xác suất lan truyền, phải nằm trong khoảng (0, 1].
        mc: int
            Số lần mô phỏng Monte Carlo, phải là số nguyên dương.

    Returns:
        bool: True nếu tất cả tham số hợp lệ, ngược lại False.
        str: Thông báo lỗi nếu có.
    """
    if not isinstance(k, int) or k <= 0:
        return False, "k phải là một số nguyên dương."

    if not (isinstance(p, float) and 0 < p <= 1):
        return False, "p phải là một số thực nằm trong khoảng (0, 1]."

    if not isinstance(mc, int) or mc <= 0:
        return False, "mc phải là một số nguyên dương."

    return True, "Các tham số đầu vào hợp lệ."


def save_graph_and_dataframe(G, df):
    # Lưu hình ảnh đồ thị
    visual_style = {
        "vertex_size": 25,
        "vertex_label": G.vs["label"],
        "vertex_label_size": 12,
        "edge_color": "#B3CDE3",
        "vertex_color": "#FBB4AE",
        "layout": G.layout("kk"),  # Kamada-Kawai layout
    }
    ig.plot(G, **visual_style, target="G_figure.png")
    df.to_csv("G_dataframe.csv", index=False)

def save_graph_result(G, output, algorithm_result):
    G.vs["color"] = "#FBB4AE"
    for seed in output[0]:
        G.vs[seed]["color"] = "#4682B4"
    layout = G.layout("kk")
    ig.plot(G, bbox=(600, 600), margin=11, layout=layout, target=f"{algorithm_result}.png")

def byte_to_decimal_string(byte_obj):
    # Chuyển byte-like object thành số nguyên
    integer_value = int.from_bytes(byte_obj, byteorder='big')  # 'big' hoặc 'little' tùy vào thứ tự byte
    # Chuyển số nguyên thành chuỗi decimal
    return str(integer_value)

def save_to_txt(filename, content):
    # Mở file với chế độ ghi ('w'), nếu file chưa tồn tại thì sẽ được tạo mới
    with open(filename, 'w') as file:
        # Ghi nội dung vào file
        file.write(content)

def computation_time_chart(celf_output, ris_output):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(ris_output[1]) + 1), ris_output[1], label="RIS", color="#FBB4AE", lw=4)
    ax.plot(range(1, len(celf_output[1]) + 1), celf_output[1], label="CELF", color="#B3CDE3", lw=4)
    ax.legend(loc=2)
    plt.ylabel('Computation Time (Seconds)')
    plt.xlabel('Size of Seed Set')
    plt.title("Thời gian tính toán")
    plt.tick_params(bottom=False, left=False)
    plt.savefig("computation_time_chart.png", format="png", dpi=300)  # Xuất đồ thị ra file PNG

def main():
    print_centered_text()
    while True:
        print("1. Tạo đồ thị mạng xã hội G")
        try:
            num_nodes = int(input("\tNhập số nút: "))
            num_edges = int(input("\tNhập số cạnh được kết nối từ mỗi nút mới: "))
        except ValueError:
            print("\t**Nhập sai định dạng đầu vào, nhập lại**")
            continue
        is_valid_G, valid_text = validate_graph_params(num_nodes, num_edges)
        if is_valid_G:
            print("\t" + valid_text)
            print("\tĐang tạo đồ thị G....")
            G, df = create_G(num_nodes, num_edges)
            show_G(G)
            save_graph_and_dataframe(G, df)
            print("\tĐã lưu đồ thị và danh sách các cạnh đồ thị")
            while True:
                try:
                    print("2. Nhập thông số đầu vào cho các thuật toán")
                    k = int(input("\tSố lượng đỉnh trong tập hạt giống: "))
                    p = float(input("\tXác suất lan truyền p (0 < p < 1): "))
                    mc = int(input("\tSố lần mô phỏng Monte Carlo: "))
                except ValueError:
                    print("\t**Nhập sai định dạng đầu vào, nhập lại**")
                    continue
                is_valid_input, valid_text = validate_inputs(k, p, mc)
                if is_valid_input:
                    print("\t" + valid_text)
                    print("\tBắt đầu chạy các thuật toán....")
                    celf_output = celf(df, k, p, mc)
                    ris_output = ris(df, k, p, mc)
                    print("3. Xuất kết quả")
                    # In ra tập seed
                    print("\tCELF Seed Set:", [int(v) for v in celf_output[0]])
                    print("\tRIS Seed Set:", [int(v) for v in ris_output[0]])
                    # Tính hàm ảnh hưởng của mỗi tập seed
                    celf_spread = IC(df, celf_output[0], p, mc)
                    ris_spread = IC(df, ris_output[0], p, mc)
                    print("\tCELF Spread:", celf_spread)
                    print("\tRIS Spread:", ris_spread)
                    # Lưu kết quả đồ thị chứa các hạt giống
                    save_to_txt("CELF_&_RIS_results", f"CELF Seed Set: {[int(v) for v in celf_output[0]]}\nRIS Seed Set: {[int(v) for v in ris_output[0]]}\nCELF Spread: {celf_spread}, RIS Spread: {ris_spread}")
                    save_graph_result(G, celf_output, "CELF_graph")
                    save_graph_result(G, ris_output, "RIS_graph")
                    computation_time_chart(celf_output, ris_output)
                    print("Đã lưu kết quả các thuật toán...")
                    break
                else:
                    print("\t" + valid_text)
                    print("\t**Nhập lại các thông số**")
            break
        else:
            print("\t" + valid_text)
            print("\t**Nhập lại các thông số**")


if __name__ == "__main__":
    main()