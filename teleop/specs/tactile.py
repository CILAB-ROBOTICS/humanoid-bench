import numpy as np

# 부위별 센서 shape 정의
tactile_data_shapes = [
    {"little_finger_tip": [3, 3]},
    {"little_finger_nail": [12, 8]},
    { "little_finger_pad": [10, 8]},
    {"ring_finger_tip": [3, 3]},
    {"ring_finger_nail": [12, 8]},
    {"ring_finger_pad": [10, 8]},
    {"middle_finger_tip": [3, 3]},
    {"middle_finger_nail": [12, 8]},
    {"middle_finger_pad": [10, 8]},
    {"index_finger_tip": [3, 3]},
    {"index_finger_nail": [12, 8]},
    {"index_finger_pad": [10, 8]},
    {"thumb_tip": [3, 3]},
    {"thumb_nail": [12, 8]},
    {"thumb_middle_section": [3, 3]},
    {"thumb_pad": [12, 8]},
    {"palm": [8, 14]}
]

def split_tactile_data(flat_array: np.ndarray) -> dict:
    """
    Splits a flat array of length 1062 into a dictionary of tactile data arrays
    according to predefined shapes.
    """
    assert flat_array.ndim == 1, "Input must be a 1D array"
    assert len(flat_array) == 1062, "Input array must have exactly 1062 elements"

    result = {}
    idx = 0
    for tactile_data_shape in tactile_data_shapes:
        key, shape = list(tactile_data_shape.items())[0]
        size = shape[0] * shape[1]
        values = flat_array[idx: idx + size]
        result[key] = values.reshape(shape)
        idx += size

    return result

def main():
    # 예제용 더미 데이터: 0 ~ 1061까지
    flat_data = np.arange(1062)

    # 분할 수행
    tactile_dict = split_tactile_data(flat_data)

    # 일부 결과 확인
    for key in list(tactile_dict.keys())[:3]:  # 처음 3개만 출력
        print(f"{key} ({tactile_dict[key].shape}):")
        print(tactile_dict[key], "\n")

if __name__ == "__main__":
    main()
