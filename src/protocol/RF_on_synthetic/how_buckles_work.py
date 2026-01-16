# # q1 = 2 q2 =3 k=10
# #     q1        q2
# #     01 234|56 789
# # A = 12 345|67 89a
# # B = 67 89a|bc def
# # C = bc def|gh ijk
# # "",

# # create: one_chunk = lastchunk[k-(q1+q2)-1:k] + points_after_last_chunk[:k-(q1+q2)] 
# # attach: attach(one_chunk, clean_traj) = clean_traj.expand(one_chunk[q1:k-q2])


# def traj2chunk():
#     q1 = 2
#     q2 = 3
#     k = 10
#     full = ["1","2","3","4","5",
#             "6","7","8","9","a",
#             "b","c","d","e","f",
#             "g","h","i","j","k"]
#     curr = 0
#     prev = []
#     chunk_lst = []
#     while curr != len(full):
#         if curr == 0:
#             dup_head = [full[0]]*q1
#             curr_lst = dup_head + full[:k-q1]
#             curr = k-q1
#         elif curr + (k-(q1+q2)) >= len(full):
#             curr_lst = prev[k-(q1+q2):k]+full[curr:]+[full[-1]]*(curr+(k-(q1+q2))-len(full))
#             curr = len(full)
#         else:
#             curr_lst = prev[k-(q1+q2):k]+full[curr:curr+(k-(q1+q2))]
#             curr += (q1+q2)
#         prev = list.copy(curr_lst)
#         chunk_lst.append(list.copy(curr_lst))
#         print(curr_lst)


# traj2chunk()

# ================================================================
# Two-buckle trajectory chunk debugger (Q1, Q2, K)
# Produces perfectly visualized chunk boundaries with `[next_start]`
# ================================================================

# ================================================================
# Two-buckle trajectory chunk debugger (Q1, Q2, K)
# Marker is placed exactly where the next chunk begins reading.
# ================================================================

def print_chunk(chunk, Q1, Q2):
    """Split into head | middle | tail and print."""
    K = len(chunk)
    head = chunk[:Q1]
    mid  = chunk[Q1:K-Q2]
    tail = chunk[K-Q2:]
    print(
        f"{' '.join(head)} | "
        f"{' '.join(mid)} | "
        f"{' '.join(tail)}"
    )


def traj2chunk_debug(full, K, Q1, Q2):
    L = len(full)
    stride = K - (Q1 + Q2)
    assert stride > 0

    chunks = []
    curr = 0

    # --------------------------
    # Chunk 0
    # --------------------------
    dup = [full[0]] * Q1
    take = full[:K-Q1]
    chunk = dup + take
    chunks.append(chunk)

    marker_idx = K - (Q1 + Q2)
    chunk_vis = list(chunk)

    if (Q1+Q2) > 0 and 0 <= marker_idx < K:
        chunk_vis[marker_idx] = f"[{chunk_vis[marker_idx]}]"

    print("=== Chunk 0 ===")
    print_chunk(chunk_vis, Q1, Q2)

    prev = chunk
    curr = K - Q1
    idx = 1

    # --------------------------
    # middle / last chunks
    # --------------------------
    while curr < L:
        prefix = prev[-(Q1+Q2):] if (Q1+Q2)>0 else []
        remain = L - curr
        need = stride

        if remain >= need:
            take = full[curr:curr+need]
            curr += need
            chunk = prefix + take
        else:
            take = full[curr:]
            pad_len = need - remain
            pad = [full[-1]] * pad_len
            chunk = prefix + take + pad
            curr = L

        chunk_vis = list(chunk)
        marker_idx = K - (Q1 + Q2)

        if (Q1+Q2)>0 and 0 <= marker_idx < K:
            chunk_vis[marker_idx] = f"[{chunk_vis[marker_idx]}]"

        print(f"=== Chunk {idx} ===")
        print_chunk(chunk_vis, Q1, Q2)

        prev = chunk
        chunks.append(chunk)
        idx += 1

    return chunks



# ================================================================
# Test case
# ================================================================
if __name__ == "__main__":
    Q1 = 2
    Q2 = 2
    K = 10

    full = ["1","2","3","4","5",
            "6","7","8","9","a",
            "b","c","d","e","f",
            "g","h","i","j","k",
            "l","m","n","o","p",
            "q","r","s","t","u",
            "v","w","x","y","z"]

    traj2chunk_debug(full, K, Q1, Q2)
