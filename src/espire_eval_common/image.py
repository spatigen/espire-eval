import base64
from pathlib import Path

import cv2
import numpy as np


def draw_points_on_image(
    image_path: str | Path, row: int, col: int, output_path: str | Path
) -> None:
    """
    Draws a red circle at specified coordinates on an image and saves the result.

    Args:
        image_path (str | Path): Path to the input image file
        row (int): Y-coordinate (row) for the point center
        col (int): X-coordinate (column) for the point center
        output_path (str | Path): Path to save the modified image

    Raises:
        IndexError: If the coordinate is out of image shape
        FileNotFoundError: If the input image path doesn't exist
    """
    # Read image with transparency support (IMREAD_UNCHANGED)
    if isinstance(image_path, Path):
        image_path = str(image_path.resolve())

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image from {image_path}")

    # Validate coordinates are within image bounds
    height, width = img.shape[:2]
    if not (0 <= col < width and 0 <= row < height):
        raise IndexError(
            f"Coordinates ({col}, {row}) are outside image dimensions ({width}x{height})"
        )

    # Draw filled red circle (BGR color format)
    cv2.circle(img, (col, row), radius=5, color=(0, 0, 255), thickness=-1)

    # Save result with optimal PNG compression
    if isinstance(output_path, Path):
        output_path = str(output_path.resolve())

    cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 5])


def encode_image(image_path: str | Path) -> str:
    """
    Encodes an image file to base64 string for web transmission or storage.

    Args:
        image_path (str | Path): Path to the image file

    Returns:
        str: Base64 encoded string of the image data

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the path is not a file or the file is empty
    """
    image_path = (
        Path(image_path) if isinstance(image_path, str) else image_path
    )  # Ensure Path object for consistency

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Path is not a file: {image_path}")

    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        if not image_data:
            raise ValueError(f"Image file is empty: {image_path}")

        return base64.b64encode(image_data).decode("utf-8")


def encode_color_png_rgb(image: np.ndarray) -> bytes:
    """
    Encodes a RGB numpy array to PNG bytes with proper color space handling.

    Args:
        image (np.ndarray): Input image array in RGB format (H,W,3) or RGBA (H,W,4)

    Returns:
        bytes: PNG encoded image data

    Raises:
        ValueError: For invalid input dimensions or data type
        RuntimeError: If PNG encoding fails

    Note:
        Converts RGB to BGR for OpenCV compatibility since OpenCV uses BGR format
    """
    image = np.asarray(image)

    # Validate input dimensions and type
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected 3D array with 3 or 4 channels, got {image.shape}")

    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 dtype, got {image.dtype}")

    # Convert RGBA to RGB if necessary (discard alpha channel)
    if image.shape[2] == 4:
        image = image[:, :, :3]  # Keep only RGB channels

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Encode to PNG with compression
    success, encoded_data = cv2.imencode(
        ".png", img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 5]
    )

    if not success:
        raise RuntimeError("Failed to encode image to PNG format")

    return encoded_data.tobytes()


def decode_color_png_rgb(png_bytes: bytes) -> np.ndarray:
    """
    Decodes PNG bytes to RGB numpy array with proper color space conversion.

    Args:
        png_bytes (bytes): PNG encoded image data

    Returns:
        np.ndarray: Decoded image in RGB format (H,W,3)

    Raises:
        RuntimeError: If decoding fails or unexpected channel format
        ValueError: If the input bytes are empty or wrong channels or dimensions
    """
    if not png_bytes:
        raise ValueError("Input bytes cannot be empty")

    # Convert bytes to numpy array and decode
    buffer = np.frombuffer(png_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise RuntimeError("Failed to decode PNG image")

    # Handle different channel configurations
    if image.ndim == 2:
        # Grayscale to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        if image.shape[2] == 3:
            # BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            # BGRA to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            raise ValueError(f"Unexpected number of channels: {image.shape[2]}")
    else:
        raise ValueError(f"Unexpected array dimensions: {image.ndim}")

    return image_rgb


def encode_depth_rawf32(depth_data: np.ndarray) -> bytes:
    """
    Encodes float32 depth map to raw bytes in row-major little-endian format.

    Args:
        depth_data (np.ndarray): Depth data in meters as float32 array

    Returns:
        bytes: Raw byte representation of the depth matrix
    """
    depth_array = np.asarray(depth_data, dtype=np.float32)
    return depth_array.tobytes()


def decode_depth_rawf32(data: bytes, row: int, col: int) -> np.ndarray:
    """
    Decode raw bytes to float32 depth map with specified dimensions.

    Args:
        data (bytes): Raw depth data bytes
        row (int): Number of rows in the depth map
        col (int): Number of columns in the depth map

    Returns:
        np.ndarray: Depth map as float32 array
    """
    return decode_mat_f32(data, row, col)


def encode_mat_f32(mat_data: np.ndarray) -> bytes:
    """
    Encodes small float32 matrices (e.g., 3x3, 4x4) to raw bytes.

    Args:
        mat_data (np.ndarray): Input matrix of float32 values

    Returns:
        bytes: Raw byte representation of mat matrix
    """
    mat_array = np.asarray(mat_data, dtype=np.float32)
    return mat_array.tobytes()


def decode_mat_f32(data: bytes, row: int, col: int) -> np.ndarray:
    """
    Decode raw bytes to float32 matrix with specified dimensions.

    Args:
        data (bytes): Raw matrix data bytes
        rows (int): Number of rows in the matrix
        cols (int): Number of columns in the matrix

    Returns:
        np.ndarray: Matrix as float32 array
    """
    array = np.frombuffer(data, dtype=np.float32)
    return array.reshape(row, col)
