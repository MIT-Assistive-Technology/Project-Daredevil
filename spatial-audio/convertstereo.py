import numpy as np
import sounddevice as sd
import threading
import time
import math

class SpatialAudioEngine:
    """
    Generates spatial audio cues based on object positions.
    Assumes camera/phone is at origin (0, 0) facing forward.
    Uses stereo panning and volume for spatial effect.
    """

    def __init__(self, sample_rate=44100, frequency=440):
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.is_running = False
        self.position = [0, 5, 0]  # Default: center, medium distance
        self.thread = None

        # Audio parameters
        self.max_distance = 20.0  # Distance at which sound is barely audible
        self.min_distance = 1.0   # Distance at which sound is at full volume

        print("Spatial Audio Engine initialized")
        print("Camera/Phone assumed at origin (0, 0, 0)")
        print(f"Sample rate: {sample_rate} Hz, Tone: {frequency} Hz")

    def _calculate_stereo_params(self, x, y, z=0):
        """
        Calculate stereo parameters based on position.

        Returns:
            left_gain: Volume for left channel (0.0 to 1.0)
            right_gain: Volume for right channel (0.0 to 1.0)
        """
        # Calculate distance
        distance = math.sqrt(x**2 + y**2 + z**2)

        # Volume based on distance (inverse square law, clamped)
        if distance < self.min_distance:
            volume = 1.0
        elif distance > self.max_distance:
            volume = 0.0
        else:
            # Smooth falloff
            volume = 1.0 - ((distance - self.min_distance) /
                           (self.max_distance - self.min_distance))
            volume = volume ** 2  # Square for more natural falloff

        # Stereo panning based on x position
        # x ranges from -10 to 10 typically, map to -1 to 1
        pan = np.clip(x / 10.0, -1.0, 1.0)

        # Calculate left and right gains using constant power panning
        # This maintains perceived loudness across the stereo field
        pan_angle = (pan + 1.0) * (math.pi / 4.0)  # Map to 0 to π/2
        left_gain = math.cos(pan_angle) * volume
        right_gain = math.sin(pan_angle) * volume

        return left_gain, right_gain

    def _generate_audio_callback(self, outdata, frames, time_info, status):
        """Callback function for sounddevice to generate audio in real-time"""
        if status:
            print(f"Audio callback status: {status}")

        # Generate time array
        t = (np.arange(frames) + self.phase) / self.sample_rate

        # Generate mono sine wave
        wave = np.sin(2 * np.pi * self.frequency * t)

        # Get current stereo parameters
        x, y, z = self.position
        left_gain, right_gain = self._calculate_stereo_params(x, y, z)

        # Apply stereo panning
        outdata[:, 0] = wave * left_gain   # Left channel
        outdata[:, 1] = wave * right_gain  # Right channel

        # Update phase for continuous playback
        self.phase += frames

    def update_object_position(self, x, y, z=0):
        """
        Update spatial audio based on object position.

        Args:
            x: Horizontal position (negative = left, positive = right)
            y: Depth/distance (negative = closer, positive = farther)
            z: Vertical position (for future use)
        """
        self.position = [x, y, z]

        # Calculate spatial parameters for display
        distance = math.sqrt(x**2 + y**2 + z**2)
        left_gain, right_gain = self._calculate_stereo_params(x, y, z)

        # Determine direction
        if x < -0.5:
            direction = "LEFT"
        elif x > 0.5:
            direction = "RIGHT"
        else:
            direction = "CENTER"

        print(f"Object at ({x:.2f}, {y:.2f}, {z:.2f}) | "
              f"Distance: {distance:.2f} | Direction: {direction} | "
              f"L:{left_gain:.2f} R:{right_gain:.2f}")

    def play(self):
        """Start playing the spatial audio"""
        if not self.is_running:
            self.is_running = True
            self.phase = 0

            # Start audio stream
            self.stream = sd.OutputStream(
                channels=2,
                callback=self._generate_audio_callback,
                samplerate=self.sample_rate,
                blocksize=1024
            )
            self.stream.start()
            print("Audio started (stereo output)")

    def stop(self):
        """Stop playing the spatial audio"""
        if self.is_running:
            self.is_running = False
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
            print("Audio stopped")

    def cleanup(self):
        """Clean up resources"""
        self.stop()
        print("Audio engine cleaned up")


def demo_spatial_audio():
    """
    Demonstration of spatial audio with various object positions.
    """
    engine = SpatialAudioEngine()

    try:
        print("\n=== Spatial Audio Demo ===")
        print("Camera is at origin (0, 0, 0)")
        print("Positive X = Right, Negative X = Left")
        print("Positive Y = Away, Negative Y = Toward\n")

        # Test scenarios
        scenarios = [
            ("Center, close", 0, 2, 0),
            ("Right side, medium distance", 4, 5, 0),
            ("Left side, medium distance", -4, 5, 0),
            ("Far right", 6, 10, 0),
            ("Far left", -6, 10, 0),
            ("Very close, slight left", -1, 1, 0),
            ("Moving left to right (close)", None),  # Animation
        ]

        engine.play()

        for item in scenarios:
            if len(item) == 4:
                description, x, y, z = item
                print(f"\n--- {description} ---")
                engine.update_object_position(x, y, z)
                time.sleep(3)
            else:
                # Animated movement
                description = item[0]
                print(f"\n--- {description} ---")
                for x in np.linspace(-5, 5, 30):
                    engine.update_object_position(x, 3, 0)
                    time.sleep(0.1)

        engine.stop()

    except KeyboardInterrupt:
        print("\nDemo interrupted")
    finally:
        engine.cleanup()


def interactive_mode():
    """
    Interactive mode: Enter coordinates to test spatial audio.
    """
    engine = SpatialAudioEngine()

    try:
        print("\n=== Interactive Spatial Audio ===")
        print("Enter object positions as: x y [z]")
        print("Example: 3 5 0  (right side, medium distance)")
        print("Type 'quit' to exit\n")

        engine.play()

        while True:
            user_input = input("Enter position (x y [z]): ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            try:
                coords = list(map(float, user_input.split()))
                x, y = coords[0], coords[1]
                z = coords[2] if len(coords) > 2 else 0

                engine.update_object_position(x, y, z)

            except (ValueError, IndexError):
                print("Invalid input. Use format: x y [z]")

        engine.stop()

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        engine.cleanup()


if __name__ == "__main__":
    print("Select mode:")
    print("1. Demo mode (automatic sequence)")
    print("2. Interactive mode (enter your own coordinates)")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        demo_spatial_audio()
    elif choice == "2":
        interactive_mode()
    else:
        print("Invalid choice. Running demo mode...")
        demo_spatial_audio()
