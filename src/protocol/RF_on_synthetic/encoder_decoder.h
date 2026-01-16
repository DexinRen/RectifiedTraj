"""
encoder_decoder_protocol.py

TRAJECTORY-WISE DENOISING PROTOCOL
-----------------------------------
This file specifies the interface for denoise_traj_DF and denoise_traj_BF.

These functions extend the chunk-wise denoising pipeline to handle
arbitrary-length GPS trajectories by orchestrating multiple chunks
with proper buckle stitching.

This is a header/protocol file:
    - NO implementations
    - Only function signatures
    - Comments describe precise required behavior
"""


# ================================================================
# === DESIGN PHILOSOPHY
# ================================================================
"""
TWO STRATEGIES FOR TRAJECTORY DENOISING:

1. DF (Depth-First): FAST DENOISING
   - Prioritize speed over accuracy
   - Each chunk denoised independently to completion
   - Trade-off: Potential discontinuities at chunk boundaries
   - Use case: Quick inference, real-time applications

2. BF (Breadth-First): ACCURATE DENOISING
   - Prioritize smoothness and accuracy over speed
   - Gradual noise reduction across all chunks simultaneously
   - Buckle sections change smoothly with neighboring points
   - Proceeding chunks receive head buckles at matching noise levels
   - Use case: Offline processing, highest quality output

KEY INSIGHT (BF):
    When chunk[i] is built at noise level t=0.8, its head buckle comes
    from chunk[i-1] that was ALSO just denoised to t=0.8.
    
    This means the buckle (connection section) provides smooth context
    that is at a SIMILAR NOISE LEVEL to the points next to it in chunk[i].
    
    In contrast, DF uses a FULLY DENOISED (t=0.0) buckle next to 
    NOISY (t=1.0) points, creating potential discontinuities.
"""


# ================================================================
# === TRAVERSAL VISUALIZATION
# ================================================================
"""
Traversal Axes:
    CHUNK AXIS (x) →     1    2    3    N
    NOISE AXIS (y) ↓
                t=1.0   [█]  [█]  [█]  [█]
                   ↓
                t=0.9   [█]  [█]  [█]  [█]
                   ↓
                t=0.8   [█]  [█]  [█]  [█]
                   ↓
                t=0.0   [█]  [█]  [█]  [█]

DF (Depth-First):  Traverse y-axis FIRST (denoise chunk fully), then x++
BF (Breadth-First): Traverse x-axis FIRST (denoise all chunks one step), then y--
"""


# ================================================================
# === CORE CONCEPTS
# ================================================================
"""
CHUNK STRUCTURE:
    CHUNK[i] = [HEAD_BUCKLE | PAYLOAD | TAIL_BUCKLE]
               [Q1 points    | middle  | Q2 points   ]
               256 total points

    stride = K - (Q1 + Q2) = 256 - 1 - 0 = 255

BUCKLE REUSE RULE:
    CHUNK[i].HEAD_BUCKLE = CHUNK[i-1].PAYLOAD[-Q1:] + CHUNK[i-1].TAIL_BUCKLE[:Q2]

COORDINATE SYSTEMS:
    - GPS: (lon, lat) — input/output format
    - ENU: (east, north) — denoising happens here
    - Reference point: First point of each chunk

DENOISING STEP:
    Vt = model(Xt, t)           # predict velocity
    Xt_next = Xt - t_delta * Vt # update position
    t_next = max(0, t - t_delta)
"""


# ================================================================
# === denoise_traj_DF (Depth-First)
# ================================================================
def denoise_traj_DF(self, traj: np.ndarray) -> dict:
    """
    Purpose:
        FAST trajectory denoising using DEPTH-FIRST traversal.
        Fully denoise each chunk before moving to the next.
        Optimized for speed at the cost of potential boundary discontinuities.

    Parameters:
        traj (np.ndarray): (T, 2) GPS trajectory [lon, lat]
            - May contain NaN rows (will be removed)
            - T is arbitrary length

    Return Dict:
        "error_code": 0 (success) | -1 (error)
        "traj_clean": (T', 2) cleaned GPS trajectory (NaN rows removed)

    Strategy:
        SPEED-OPTIMIZED DENOISING
        - Each chunk processed independently to completion
        - No inter-chunk synchronization
        - Trade-off: Head buckle at t=0.0 connects to payload at t=1.0
          (potential noise level mismatch at boundaries)

    Traversal Order:
        FOR x in [1, 2, ..., N]:     # outer loop: chunks
            FOR y in [1.0, 0.9, ..., 0.0]:  # inner loop: noise levels
                denoise_one_step(chunk[x], y)

    ASCII Workflow:
        CHUNK 1: t=1.0 → t=0.8 → ... → t=0.0 ✓ (DONE)
                    ↓
        CHUNK 2:    t=1.0 → t=0.8 → ... → t=0.0 ✓ (uses chunk 1's t=0.0 tail)
                        ↓
        CHUNK 3:        t=1.0 → ... → t=0.0 ✓ (uses chunk 2's t=0.0 tail)

        DIRECTION: y-axis first (denoise fully), then x-axis (next chunk)

    Algorithm:
        1. Remove NaN rows from input trajectory
        2. Build chunk[0] with duplicated head buckle
        3. FOR i in range(num_chunks):
            a. IF i > 0:
                - Extract head buckle from FULLY DENOISED chunk[i-1]
                - Concatenate with remaining trajectory points
            b. Transform chunk to ENU (using chunk[0] as origin)
            c. FULLY denoise chunk (t: 1.0 → 0.0)
            d. Transform back to GPS
            e. Extract payload (strip Q1 head, Q2 tail)
            f. Append payload to output
        4. Return stitched trajectory

    Key Characteristics:
        - Each chunk completely denoised before next
        - Chunk boundaries use fully-denoised previous chunk's tail
        - Simple, sequential processing
        - Memory: O(K) active chunk only
        - FASTEST denoising mode

    Critical Dependency:
        chunk[i] uses chunk[i-1]'s FINAL (t=0.0) tail as head buckle
        
        BOUNDARY CONDITION:
            HEAD_BUCKLE (t=0.0) | PAYLOAD (t=1.0)
                                ↑
                    Potential discontinuity

    Usage:
        Called by EncoderDecoder.denoise_traj() when user selects DF mode.
        Recommended for: real-time applications, quick inference.
    """
    pass


# ================================================================
# === denoise_traj_BF (Breadth-First)
# ================================================================
def denoise_traj_BF(self, traj: np.ndarray) -> dict:
    """
    Purpose:
        ACCURATE trajectory denoising using BREADTH-FIRST traversal.
        Synchronize noise reduction across all chunks for maximum smoothness.
        Optimized for quality at the cost of higher memory usage.

    Parameters:
        traj (np.ndarray): (T, 2) GPS trajectory [lon, lat]
            - May contain NaN rows (will be removed)
            - T is arbitrary length

    Return Dict:
        "error_code": 0 (success) | -1 (error)
        "traj_clean": (T', 2) cleaned GPS trajectory (NaN rows removed)

    Strategy:
        ACCURACY-OPTIMIZED DENOISING
        - Gradual noise reduction across entire trajectory
        - Chunk boundaries receive buckles at matching noise levels
        - Buckle sections change smoothly with neighboring points
        - Proceeding chunks get context that is close to their noise level

    Traversal Order:
        FOR y in [1.0, 0.9, ..., 0.0]:  # outer loop: noise levels
            FOR x in [1, 2, ..., N]:     # inner loop: chunks
                denoise_one_step(chunk[x], y)

    ASCII Workflow:
        NOISE LEVEL (y) ↓

        t=1.0:  chunk1 → chunk2 → chunk3 → ... → chunkN  (all one step)
                 ↓        ↓        ↓               ↓
               t=0.9    t=0.9    t=0.9           t=0.9

        t=0.9:  chunk1 → chunk2 → chunk3 → ... → chunkN  (all one step)
                 ↓        ↓        ↓               ↓
               t=0.8    t=0.8    t=0.8           t=0.8

        ...

        t=0.0:  [final stitched trajectory]

        DIRECTION: x-axis first (denoise all chunks one step), then y-axis (decrease noise)

    Algorithm:
        1. Remove NaN rows from input trajectory
        2. Initialize: trajectories[1.0] = original_gps_traj
        3. t_current = 1.0
        4. WHILE t_current > 0:
            a. t_next = max(0, t_current - t_delta)
            b. denoised_chunks_full = []  # store FULL chunks (with buckles)
            c. payloads = []               # store only payloads
            
            d. FOR i in range(num_chunks):
                # Build chunk[i] at noise level t_current
                IF i == 0:
                    - Build first chunk with duplicated head buckle
                ELSE:
                    - Extract head buckle from chunk[i-1] (JUST denoised to t_next)
                    - Concatenate with remaining points from trajectories[t_current]
                
                # Transform to ENU
                - chunk_enu, origin = gps_to_enu(chunk_gps)
                
                # Denoise ONE STEP: t_current → t_next
                - Vt = model(chunk_enu, t_current)
                - chunk_enu_next = chunk_enu - t_delta * Vt
                
                # Transform back to GPS
                - chunk_gps_next = enu_to_gps(chunk_enu_next, origin)
                
                # Store full chunk and payload
                - denoised_chunks_full.append(chunk_gps_next)
                - payload = chunk_gps_next[Q1 : K-Q2]
                - payloads.append(payload)
            
            e. trajectories[t_next] = stitch(payloads)
            f. t_current = t_next
        
        5. Return trajectories[0.0]

    Key Characteristics:
        - All chunks denoised one step per noise level
        - Chunk boundaries continuously updated at each noise level
        - Better inter-chunk smoothness (synchronized noise reduction)
        - Memory: O(T) per noise level
        - HIGHEST QUALITY denoising mode

    Critical Dependency:
        chunk[i] at t_current uses chunk[i-1]'s PARTIALLY DENOISED tail
        (the one we JUST denoised from t_current → t_next)

        SMOOTH BOUNDARY CONDITION:
            HEAD_BUCKLE (t=0.8) | PAYLOAD (t=0.9)
                                ↑
                    Smooth transition (similar noise levels)

        This creates smoother chunk boundaries because:
        1. Buckle and payload have similar noise levels
        2. Gradual noise reduction prevents abrupt changes
        3. Head buckle provides smooth context to proceeding chunk

    Why BF is More Accurate:
        When chunk[i] is built at noise level t=0.8:
        - Its head buckle comes from chunk[i-1] at t=0.8
        - The buckle (connection section) does not change too much
          compared to the next un-denoised section
        - This provides smoother context that is CLOSE to the noise
          level of the points next to them in chunk[i]

    Index Calculation:
        # BF: chunk[i] uses chunk[i-1]'s PARTIALLY DENOISED tail
        head_buckle = denoised_chunks_full[i-1][K-(Q1+Q2) : K-Q2]  # last Q1
        tail_buckle = denoised_chunks_full[i-1][K-Q2 : K]          # last Q2
        buckle = concat(head_buckle, tail_buckle)

    Usage:
        Called by EncoderDecoder.denoise_traj() when user selects BF mode.
        Recommended for: offline processing, highest quality output.
    """
    pass


# ================================================================
# === COMPARISON TABLE
# ================================================================
"""
| Aspect                  | DF (Depth-First)       | BF (Breadth-First)      |
|-------------------------|------------------------|-------------------------|
| Optimization Goal       | Speed                  | Accuracy                |
| Outer Loop              | Chunks (x)             | Noise levels (y)        |
| Inner Loop              | Noise levels (y)       | Chunks (x)              |
| Traversal Direction     | y-first (vertical)     | x-first (horizontal)    |
| Chunk[i] Buckle Source  | chunk[i-1] at t=0.0    | chunk[i-1] at t_current |
| Buckle-Payload Match    | Mismatched noise       | Matched noise levels    |
| Boundary Smoothness     | Potential discontinuity| Smooth (synchronized)   |
| Parallelization         | Chunks independent     | Sequential per noise    |
| Memory Usage            | O(K) active chunk      | O(T) per noise level    |
| Speed                   | FASTEST                | Slower (more iterations)|
| Quality                 | Good                   | HIGHEST                 |
| Use Case                | Real-time, quick       | Offline, quality-first  |
"""


# ================================================================
# === BUCKLE MECHANICS (CRITICAL)
# ================================================================
"""
Index Calculation (from encode-decode.py):

    # Chunk 0
    head_buckle = duplicate(traj[0], Q1)  # repeat first point Q1 times
    payload_points = traj[0 : K-Q1]
    chunk_0 = concat(head_buckle, payload_points)

    # Chunk i (i > 0)
    head_buckle = prev_chunk[K-(Q1+Q2) : K]  # last Q1+Q2 points
    payload_points = traj[curr : curr+stride]
    chunk_i = concat(head_buckle, payload_points)

    # After denoising
    payload = chunk_clean[Q1 : K-Q2]  # strip head/tail buckles

Connection Points:

    CHUNK[i-1]:  [...  X  X  X] ← last Q1+Q2 points
                        ↑  ↑  ↑
    CHUNK[i]:    [X  X  X  ...] ← first Q1+Q2 points (head buckle)

Noise Level Matching (BF Advantage):

    DF:  chunk[i-1] at t=0.0  →  chunk[i] at t=1.0
         [CLEAN BUCKLE] | [NOISY PAYLOAD]
                        ↑
                Noise level jump

    BF:  chunk[i-1] at t=0.8  →  chunk[i] at t=0.9
         [t=0.8 BUCKLE] | [t=0.9 PAYLOAD]
                        ↑
                Smooth transition
"""


# ================================================================
# === ASCII GRID VISUALIZATION
# ================================================================
"""
        CHUNK AXIS (x) →
        1    2    3    4
1.0    [→]  [→]  [→]  [█]   BF: horizontal scan
0.9    [→]  [→]  [→]  [█]   (process all chunks at each noise level)
0.8    [→]  [→]  [→]  [█]   → Better boundary smoothness
  ↓
0.0    [█]  [█]  [█]  [█]

VS

        1    2    3    4
1.0    [↓]  [█]  [█]  [█]   DF: vertical scan
0.9    [↓]  [█]  [█]  [█]   (fully denoise each chunk)
0.8    [↓]  [█]  [█]  [█]   → Faster processing
  ↓     ↓
0.0    [✓]  [↓]  [█]  [█]
"""


# ================================================================
# END OF PROTOCOL
# ================================================================