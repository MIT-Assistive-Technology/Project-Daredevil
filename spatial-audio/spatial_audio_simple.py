#!/usr/bin/env python3
"""
Simplified Spatial Audio - WORKING VERSION
Uses PyGame for 3D spatial audio that actually works
"""

import pygame
import numpy as np
import threading
import time
import os
from typing import Dict, Optional, Tuple, List


class SimpleSpatialAudio:
    """Simplified spatial audio using PyGame's mixer (actually works!)"""

    def __init__(
        self, max_sources=10, master_volume=0.1, min_volume=0.02, max_volume=0.3, sounds_dir=None, debug=False
    ):
        self.max_sources = max_sources
        self.master_volume = master_volume  # Base volume level
        self.min_volume = min_volume  # Minimum volume when no objects detected
        self.max_volume = max_volume  # Maximum volume for very close objects
        self.debug = debug
        self.objects: Dict[str, dict] = {}
        self.object_channels: Dict[str, int] = (
            {}
        )  # Track which channel each object uses
        self.background_sources: Dict[str, dict] = {}  # Background depth noise sources
        self.background_channels: Dict[str, int] = {}  # Channels for background sources
        self.lock = threading.Lock()
        self.is_running = False
        self.ambient_volume_ratio = 0.25  # Background noise is 25% of min_volume

        # Determine sounds directory (default to parent/sounds)
        if sounds_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            sounds_dir = os.path.join(parent_dir, "sounds")
        self.sounds_dir = sounds_dir

        # Initialize pygame mixer first
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            # Verify mixer is initialized
            if not pygame.mixer.get_init():
                raise RuntimeError("Pygame mixer failed to initialize")
            
            # Reserve enough channels for audio sources
            # pygame.mixer.set_num_channels() sets the number of available channels
            pygame.mixer.set_num_channels(max(16, self.max_sources + 2))  # Reserve extra channels
            actual_channels = pygame.mixer.get_num_channels()
            print("✓ PyGame Audio initialized successfully")
            print(f"Mixer channels available: {actual_channels}")
            if actual_channels < self.max_sources:
                print(f"WARNING: Only {actual_channels} channels available, but {self.max_sources} max sources requested")
        except Exception as e:
            print(f"ERROR: Failed to initialize pygame mixer: {e}")
            raise

        # Load sound files
        self.sound_cache: Dict[str, Optional[pygame.mixer.Sound]] = {}
        self._load_sound_files()

        print(f"Max sources: {max_sources}")
        print(f"Master volume: {master_volume} (base level)")
        print(f"Volume range: {min_volume} - {max_volume}")

    def _load_sound_files(self):
        """Load MP3 sound files for specific classes."""
        # Map classes to sound files
        sound_mappings = {
            "person": "person.mp3",
            "bottle": "water.mp3",
            "cup": "water.mp3",
        }

        for class_name, filename in sound_mappings.items():
            sound_path = os.path.join(self.sounds_dir, filename)
            if os.path.exists(sound_path):
                try:
                    self.sound_cache[class_name] = pygame.mixer.Sound(sound_path)
                    print(f"✓ Loaded sound for {class_name}: {filename}")
                except Exception as e:
                    print(f"⚠ Failed to load {sound_path}: {e}")
                    self.sound_cache[class_name] = None
            else:
                print(f"⚠ Sound file not found: {sound_path}")
                self.sound_cache[class_name] = None

    def _get_sound_type_for_class(self, class_name: str) -> str:
        """Determine sound type based on class name."""
        class_lower = class_name.lower()
        
        # Map specific classes to their sounds
        if class_lower == "person":
            return "person"
        elif class_lower in ["bottle", "cup", "water", "water bottle"]:
            return "water"
        else:
            # Default to white noise for environmental objects (tables, hallways, etc.)
            return "white_noise"

    def _generate_white_noise(self, duration=0.5, pitch=440.0):
        """Generate calm white noise with optional pitch variation"""
        sample_rate = 44100
        num_samples = int(duration * sample_rate)

        # Generate calm white noise - make it quieter and more subtle
        noise = np.random.normal(0, 0.08, (num_samples, 2)).astype(np.float32)

        # Heavy low-pass filter for calmness (smoothing)
        window_size = 20  # Larger window for more smoothing
        if num_samples > window_size:
            kernel = np.ones(window_size) / window_size
            noise[:, 0] = np.convolve(noise[:, 0], kernel, mode="same")
            noise[:, 1] = np.convolve(noise[:, 1], kernel, mode="same")

        # Add VERY subtle pitch variation based on depth (not a beep!)
        # This creates a slight frequency shift rather than a tone
        t = np.linspace(0, duration, num_samples)
        # Very gentle modulation - not a beep, just slight frequency variation
        pitch_modulation = 0.02 * np.sin(
            2 * np.pi * pitch * 0.01 * t
        )  # Very slow modulation
        noise[:, 0] += pitch_modulation
        noise[:, 1] += pitch_modulation

        # Convert to int16
        noise_int16 = (noise * 32767).astype(np.int16)

        return noise_int16

    def _screen_to_pan(self, x, y, depth):
        """Convert screen position to stereo pan"""
        # X position controls left/right panning (-1.0 to 1.0)
        pan = (x - 0.5) * 2.0  # 0.0-1.0 -> -1.0 to 1.0

        # Depth controls volume (closer = louder, farther = quieter)
        # depth 0.0 = close (loud), 1.0 = far (very quiet)
        # Use exponential curve for better distance perception
        volume_multiplier = (1.0 - depth) ** 2  # Quadratic falloff

        return pan, volume_multiplier

    def _calculate_dynamic_volume(self, base_volume, depth, num_objects):
        """
        Calculate dynamic volume based on distance and number of objects.

        Args:
            base_volume: Base volume from object
            depth: Depth value (0.0 = close, 1.0 = far)
            num_objects: Number of currently detected objects

        Returns:
            Final volume level
        """
        # If no objects detected, use minimum volume
        if num_objects == 0:
            return self.min_volume

        # Calculate distance-based volume scaling
        # Closer objects (lower depth) = louder
        distance_factor = (
            1.0 - depth
        ) ** 1.5  # Slightly less aggressive than quadratic

        # Scale volume based on distance
        # Close objects (depth=0.0) get max volume, far objects (depth=1.0) get min volume
        volume_range = self.max_volume - self.min_volume
        distance_volume = self.min_volume + (volume_range * distance_factor)

        # Apply base volume scaling
        final_volume = distance_volume * base_volume

        # Clamp to volume range
        final_volume = max(self.min_volume, min(self.max_volume, final_volume))

        return final_volume

    def _depth_to_pitch(self, depth):
        """Convert depth to subtle pitch variation (closer = slightly higher)"""
        # depth 0.0 (close) -> slightly higher (500 Hz)
        # depth 1.0 (far) -> slightly lower (300 Hz)
        # Small range for subtle variation, not a beep
        pitch_hz = 300 + (500 - 300) * (1.0 - depth)
        return pitch_hz

    def _y_to_pitch_modulation(self, y):
        """Convert Y position to vertical pitch shift (up = slightly higher pitch)"""
        # y 0.0 (top) -> slightly higher pitch
        # y 1.0 (bottom) -> slightly lower pitch
        # Very subtle - maybe 50 Hz range
        pitch_offset = 50 * (0.5 - y)  # Center (0.5) = no offset
        return pitch_offset

    def update_object(self, object_id, x, y, depth, volume=None, active=True, class_name=None):
        """Update object position"""
        with self.lock:
            # Check if this is a new object or updated position
            is_new_object = object_id not in self.objects

            prev_obj = self.objects.get(object_id, {})
            volume_val = volume if volume else self.master_volume

            # Determine sound type based on class
            sound_type = self._get_sound_type_for_class(class_name or "unknown")

            # Check if position/depth/volume/class changed significantly
            position_changed = is_new_object or (
                abs(prev_obj.get("x", 0) - x) > 0.05
                or abs(prev_obj.get("y", 0) - y) > 0.05
                or abs(prev_obj.get("depth", 0) - depth) > 0.1
                or abs(prev_obj.get("volume", 0) - volume_val) > 0.05
                or prev_obj.get("sound_type", "") != sound_type
            )

            self.objects[object_id] = {
                "x": x,
                "y": y,
                "depth": depth,
                "volume": volume_val,
                "active": active,
                "class_name": class_name,
                "sound_type": sound_type,
                "last_update": time.time(),
                "needs_new_sound": position_changed,  # Flag to indicate if sound needs regenerating
            }
            
            if self.debug and (is_new_object or position_changed):
                print(f"DEBUG: Updated object {object_id}: class={class_name}, sound_type={sound_type}, active={active}, depth={depth:.2f}, volume={volume_val:.2f}, needs_sound={position_changed}")

    def _play_object_sound(self, object_id, obj_data):
        """Play spatial sound for object"""
        if not obj_data["active"]:
            if self.debug:
                print(f"DEBUG: Object {object_id} is not active, skipping")
            return

        # Only play new sound if position/depth/volume changed significantly
        if not obj_data.get("needs_new_sound", True):
            if self.debug:
                print(f"DEBUG: Object {object_id} doesn't need new sound")
            return

        # Mark that we've played a sound for this position
        obj_data["needs_new_sound"] = False

        # Calculate panning and pitch based on depth and position
        pan, volume_mult = self._screen_to_pan(
            obj_data["x"], obj_data["y"], obj_data["depth"]
        )
        base_pitch = self._depth_to_pitch(obj_data["depth"])
        y_pitch_offset = self._y_to_pitch_modulation(obj_data["y"])
        final_pitch = base_pitch + y_pitch_offset

        # Calculate dynamic volume based on distance and number of objects
        num_objects = len([o for o in self.objects.values() if o["active"]])
        final_volume = self._calculate_dynamic_volume(
            obj_data["volume"], obj_data["depth"], num_objects
        )

        # If object is too far or volume too low, don't play
        if final_volume < self.min_volume * 0.5:  # More lenient threshold
            if self.debug:
                print(f"DEBUG: Object {object_id} volume too low: {final_volume} < {self.min_volume * 0.5}")
            return

        sound_type = obj_data.get("sound_type", "white_noise")
        class_name = obj_data.get("class_name", "unknown")

        # Get or assign channel for this object
        if object_id in self.object_channels:
            channel_idx = self.object_channels[object_id]
        else:
            # Use a simple modulo to get channel index, but ensure it's within available channels
            max_channels = pygame.mixer.get_num_channels()
            if max_channels == 0:
                if self.debug:
                    print(f"ERROR: No channels available!")
                return
            channel_idx = abs(hash(object_id)) % min(self.max_sources, max_channels)
            self.object_channels[object_id] = channel_idx
            if self.debug:
                print(f"DEBUG: Created channel {channel_idx} for object {object_id}")

        try:
            channel = pygame.mixer.Channel(channel_idx)
            if channel is None:
                if self.debug:
                    print(f"ERROR: Failed to get channel {channel_idx}")
                return

            # IMPORTANT: Stop the channel before playing a new sound
            # This prevents accumulating sounds
            channel.stop()

            # Use class-specific sound if available, otherwise use white noise
            # Check if we have a cached sound for this class
            if class_name and class_name.lower() in self.sound_cache:
                cached_sound = self.sound_cache[class_name.lower()]
                if cached_sound is not None:
                    # Use the loaded MP3 sound
                    sound = cached_sound
                    
                    # Calculate stereo panning volumes
                    left_vol = max(0, min(1, (1.0 - pan) / 2.0))
                    right_vol = max(0, min(1, (1.0 + pan) / 2.0))
                    
                    # Set volume and panning on channel
                    # pygame Channel.set_volume takes (left_volume, right_volume)
                    channel.set_volume(final_volume * left_vol, final_volume * right_vol)
                    channel.play(sound, loops=-1)
                    
                    if self.debug:
                        print(f"DEBUG: Playing {class_name} sound for {object_id} on channel {channel_idx}, volume={final_volume:.2f}, pan={pan:.2f}")
                    return
                # Fall through to white noise if loading failed
                if self.debug:
                    print(f"DEBUG: Cached sound for {class_name} is None, using white noise")
            else:
                if self.debug:
                    print(f"DEBUG: No cached sound for {class_name}, using white noise")

            # Use white noise for environmental objects or if specific sound not available
            # Generate noise with pitch variation based on depth and vertical position
            noise = self._generate_white_noise(duration=0.5, pitch=final_pitch)

            # Apply stereo panning (left/right positioning)
            left_vol = max(0, min(1, (1.0 - pan) / 2.0))
            right_vol = max(0, min(1, (1.0 + pan) / 2.0))

            # Adjust volume per channel with panning
            noise[:, 0] = (noise[:, 0] * left_vol * final_volume).astype(np.int16)
            noise[:, 1] = (noise[:, 1] * right_vol * final_volume).astype(np.int16)

            # Convert to pygame sound
            sound = pygame.sndarray.make_sound(noise)
            channel.play(sound, loops=-1)  # Loop infinitely
            
            if self.debug:
                print(f"DEBUG: Playing white noise for {object_id} on channel {channel_idx}, volume={final_volume:.2f}, pan={pan:.2f}")

        except Exception as e:
            # Don't silently fail - print error for debugging
            print(f"ERROR playing sound for {object_id}: {e}")
            import traceback
            if self.debug:
                traceback.print_exc()

    def update_background_depth(self, depth_samples: List[Dict[str, float]]):
        """
        Update background depth-based ambient noise for areas without detected objects.
        
        Args:
            depth_samples: List of dicts with keys 'x', 'y', 'depth' for background areas
                          These should be sampled from depth map excluding object regions
        """
        with self.lock:
            # Limit number of background sources to avoid overwhelming
            max_background_sources = min(8, self.max_sources // 2)
            
            # Create or update background sources
            current_bg_ids = set()
            
            # Sample or use provided depth samples (limit to max_background_sources)
            if len(depth_samples) > max_background_sources:
                # Sample evenly across the list
                step = len(depth_samples) // max_background_sources
                depth_samples = depth_samples[::step][:max_background_sources]
            
            for idx, sample in enumerate(depth_samples[:max_background_sources]):
                bg_id = f"bg_depth_{idx}"
                current_bg_ids.add(bg_id)
                
                x = sample.get("x", 0.5)
                y = sample.get("y", 0.5)
                depth = sample.get("depth", 0.5)
                
                # Check if this background source already exists
                is_new = bg_id not in self.background_sources
                prev_bg = self.background_sources.get(bg_id, {})
                
                # Check if position changed significantly
                position_changed = is_new or (
                    abs(prev_bg.get("x", 0) - x) > 0.15  # Less sensitive for background
                    or abs(prev_bg.get("y", 0) - y) > 0.15
                    or abs(prev_bg.get("depth", 0) - depth) > 0.2
                )
                
                self.background_sources[bg_id] = {
                    "x": x,
                    "y": y,
                    "depth": depth,
                    "volume": self.min_volume * self.ambient_volume_ratio,  # Very quiet
                    "active": True,
                    "last_update": time.time(),
                    "needs_new_sound": position_changed,
                }
            
            # Remove old background sources that are no longer in the list
            for bg_id in list(self.background_sources.keys()):
                if bg_id not in current_bg_ids:
                    # Remove after timeout
                    if time.time() - self.background_sources[bg_id]["last_update"] > 1.0:
                        if bg_id in self.background_channels:
                            channel_idx = self.background_channels[bg_id]
                            try:
                                channel = pygame.mixer.Channel(channel_idx)
                                channel.stop()
                            except:
                                pass
                            del self.background_channels[bg_id]
                        del self.background_sources[bg_id]
    
    def _play_background_sound(self, bg_id, bg_data):
        """Play quiet ambient white noise for background depth areas"""
        if not bg_data["active"]:
            return
        
        # Only play new sound if position changed significantly
        if not bg_data.get("needs_new_sound", True):
            return
        
        bg_data["needs_new_sound"] = False
        
        # Calculate panning and pitch
        pan, _ = self._screen_to_pan(bg_data["x"], bg_data["y"], bg_data["depth"])
        base_pitch = self._depth_to_pitch(bg_data["depth"])
        y_pitch_offset = self._y_to_pitch_modulation(bg_data["y"])
        final_pitch = base_pitch + y_pitch_offset
        
        # Background volume is much quieter - use ambient_volume_ratio
        final_volume = bg_data["volume"]  # Already set to min_volume * ambient_volume_ratio
        
        # Get or assign channel for this background source
        if bg_id in self.background_channels:
            channel_idx = self.background_channels[bg_id]
        else:
            max_channels = pygame.mixer.get_num_channels()
            if max_channels == 0:
                return
            # Use higher channel indices for background (avoid conflicts with object channels)
            channel_idx = (abs(hash(bg_id)) % (max_channels - self.max_sources)) + self.max_sources
            channel_idx = min(channel_idx, max_channels - 1)
            self.background_channels[bg_id] = channel_idx
        
        try:
            channel = pygame.mixer.Channel(channel_idx)
            if channel is None:
                return
            
            channel.stop()
            
            # Generate quiet white noise for background
            noise = self._generate_white_noise(duration=0.5, pitch=final_pitch)
            
            # Apply stereo panning
            left_vol = max(0, min(1, (1.0 - pan) / 2.0))
            right_vol = max(0, min(1, (1.0 + pan) / 2.0))
            
            # Adjust volume per channel with panning (much quieter than objects)
            noise[:, 0] = (noise[:, 0] * left_vol * final_volume).astype(np.int16)
            noise[:, 1] = (noise[:, 1] * right_vol * final_volume).astype(np.int16)
            
            # Convert to pygame sound
            sound = pygame.sndarray.make_sound(noise)
            channel.play(sound, loops=-1)
            
            if self.debug:
                print(f"DEBUG: Playing background noise for {bg_id} on channel {channel_idx}, volume={final_volume:.4f}")
        
        except Exception as e:
            if self.debug:
                print(f"ERROR playing background sound for {bg_id}: {e}")

    def _update_loop(self):
        """Update loop"""
        while self.is_running:
            try:
                with self.lock:
                    current_time = time.time()

                    # Track currently active object IDs
                    current_active_ids = set()
                    active_objects = []

                    for object_id, obj_data in list(self.objects.items()):
                        # Remove old objects that haven't been updated in 2 seconds
                        if current_time - obj_data["last_update"] > 2.0:
                            # Stop the channel for this removed object
                            if object_id in self.object_channels:
                                channel_idx = self.object_channels[object_id]
                                try:
                                    channel = pygame.mixer.Channel(channel_idx)
                                    channel.stop()
                                except:
                                    pass
                                del self.object_channels[object_id]
                            del self.objects[object_id]
                            continue

                        if obj_data["active"]:
                            current_active_ids.add(object_id)
                            active_objects.append((object_id, obj_data))

                    # Stop channels for objects that are no longer active
                    for object_id in list(self.object_channels.keys()):
                        if object_id not in current_active_ids:
                            channel_idx = self.object_channels[object_id]
                            try:
                                channel = pygame.mixer.Channel(channel_idx)
                                channel.stop()
                            except:
                                pass
                            del self.object_channels[object_id]

                    # Play audio for currently active objects
                    if len(active_objects) > 0:
                        if self.debug:
                            print(f"DEBUG: {len(active_objects)} active objects, playing sounds")
                        for object_id, obj_data in active_objects:
                            self._play_object_sound(object_id, obj_data)
                    elif self.debug:
                        print("DEBUG: No active objects to play")
                    
                    # Play background depth noise for environmental areas
                    active_background = [(bg_id, bg_data) for bg_id, bg_data in self.background_sources.items() 
                                       if bg_data["active"]]
                    if len(active_background) > 0:
                        for bg_id, bg_data in active_background:
                            self._play_background_sound(bg_id, bg_data)

                time.sleep(0.1)  # Update every 100ms

            except Exception as e:
                print(f"Error in update loop: {e}")
                time.sleep(0.1)

    def start(self):
        """Start audio system"""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print("Spatial audio started")

    def stop(self):
        """Stop audio system"""
        self.is_running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        pygame.mixer.quit()
        print("Spatial audio stopped")

    def remove_object(self, object_id):
        """Remove object and stop its sound"""
        with self.lock:
            # Stop the channel for this object
            if object_id in self.object_channels:
                channel_idx = self.object_channels[object_id]
                try:
                    channel = pygame.mixer.Channel(channel_idx)
                    channel.stop()
                except:
                    pass
                # Remove from tracking
                del self.object_channels[object_id]

            # Remove from objects dict
            if object_id in self.objects:
                del self.objects[object_id]

    def get_stats(self):
        """Get engine statistics"""
        with self.lock:
            channels_in_use = len(self.object_channels)
            total_objects = len(self.objects)
            active_sources = len([o for o in self.objects.values() if o["active"]])
        
        # Check channel busy status outside lock to avoid deadlock
        active_channels = 0
        try:
            for ch_idx in list(self.object_channels.values()):
                try:
                    channel = pygame.mixer.Channel(ch_idx)
                    if channel.get_busy():
                        active_channels += 1
                except:
                    pass
        except:
            pass
        
        return {
            "total_objects": total_objects,
            "active_sources": active_sources,
            "active_channels": active_channels,
            "max_sources": self.max_sources,
            "master_volume": self.master_volume,
            "channels_in_use": channels_in_use,
        }
    
    def test_channel(self, channel_idx=0):
        """Test if a specific channel can play sound"""
        try:
            channel = pygame.mixer.Channel(channel_idx)
            test_noise = self._generate_white_noise(duration=0.1, pitch=440.0)
            test_sound = pygame.sndarray.make_sound(test_noise)
            channel.play(test_sound)
            time.sleep(0.2)
            is_busy = channel.get_busy()
            channel.stop()
            return is_busy
        except Exception as e:
            print(f"ERROR testing channel {channel_idx}: {e}")
            return False


def test():
    """Test the spatial audio"""
    print("Testing Simplified Spatial Audio...")
    print("=" * 60)

    audio = SimpleSpatialAudio(
        max_sources=10,
        master_volume=0.1,  # Base volume level
        min_volume=0.02,  # Very quiet when no objects
        max_volume=0.3,  # Louder for close objects
    )

    try:
        audio.start()
        print("✓ Audio started")

        print("\nTest 1: Object on left")
        audio.update_object("obj1", x=0.2, y=0.5, depth=0.5)
        time.sleep(3)

        print("Test 2: Object on right")
        audio.update_object("obj2", x=0.8, y=0.5, depth=0.5)
        time.sleep(3)

        print("Test 3: Move object to center")
        audio.update_object("obj1", x=0.5, y=0.5, depth=0.5)
        time.sleep(2)

        print("Test 4: Objects getting closer (louder)")
        audio.update_object("obj1", x=0.3, y=0.3, depth=0.2)
        audio.update_object("obj2", x=0.7, y=0.3, depth=0.2)
        time.sleep(3)

        print("\n✓ All tests complete!")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        audio.stop()


if __name__ == "__main__":
    test()
