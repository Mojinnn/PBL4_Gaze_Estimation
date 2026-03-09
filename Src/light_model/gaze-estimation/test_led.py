#!/usr/bin/env python3
"""
Script test đơn giản để kiểm tra LED connections
"""

import time
import sys

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("Error: RPi.GPIO not found!")
    print("Install with: pip3 install RPi.GPIO")
    sys.exit(1)

# Cấu hình GPIO pins
LED_PINS = {
    'left': 17,    # GPIO 17 - Pin 11
    'right': 27,   # GPIO 27 - Pin 13
    'up': 22,      # GPIO 22 - Pin 15
    'down': 23     # GPIO 23 - Pin 16
}

def setup_gpio():
    """Khởi tạo GPIO"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    for direction, pin in LED_PINS.items():
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    
    print("✓ GPIO initialized")

def cleanup_gpio():
    """Dọn dẹp GPIO"""
    for pin in LED_PINS.values():
        GPIO.output(pin, GPIO.LOW)
    GPIO.cleanup()
    print("\n✓ GPIO cleaned up")

def test_individual_leds():
    """Test từng LED riêng lẻ"""
    print("\n" + "="*50)
    print("  Testing individual LEDs")
    print("="*50)
    
    for direction, pin in LED_PINS.items():
        print(f"\nTesting {direction.upper():>6} LED (GPIO {pin:>2})...", end=' ')
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(pin, GPIO.LOW)
        print("✓")
        time.sleep(0.3)

def test_all_leds_together():
    """Test tất cả LED cùng lúc"""
    print("\n" + "="*50)
    print("  Testing all LEDs together")
    print("="*50)
    
    print("\nTurning ON all LEDs...", end=' ')
    for pin in LED_PINS.values():
        GPIO.output(pin, GPIO.HIGH)
    print("✓")
    time.sleep(2)
    
    print("Turning OFF all LEDs...", end=' ')
    for pin in LED_PINS.values():
        GPIO.output(pin, GPIO.LOW)
    print("✓")

def test_pattern():
    """Test pattern: trái -> phải -> lên -> xuống"""
    print("\n" + "="*50)
    print("  Testing directional pattern")
    print("="*50)
    
    directions = ['left', 'right', 'up', 'down']
    
    print("\nPattern: LEFT → RIGHT → UP → DOWN")
    for _ in range(3):  # Lặp 3 lần
        for direction in directions:
            pin = LED_PINS[direction]
            GPIO.output(pin, GPIO.HIGH)
            print(f"  → {direction.upper()}", end='', flush=True)
            time.sleep(0.5)
            GPIO.output(pin, GPIO.LOW)
            time.sleep(0.2)
        print()

def test_blink():
    """Test nhấp nháy tất cả LED"""
    print("\n" + "="*50)
    print("  Testing LED blink")
    print("="*50)
    
    print("\nBlinking all LEDs 5 times...")
    for i in range(5):
        # ON
        for pin in LED_PINS.values():
            GPIO.output(pin, GPIO.HIGH)
        print(f"  Blink {i+1}/5: ON ", end='', flush=True)
        time.sleep(0.3)
        
        # OFF
        for pin in LED_PINS.values():
            GPIO.output(pin, GPIO.LOW)
        print("OFF")
        time.sleep(0.3)

def interactive_test():
    """Test tương tác - điều khiển LED bằng bàn phím"""
    print("\n" + "="*50)
    print("  Interactive LED Control")
    print("="*50)
    print("\nControls:")
    print("  a/A - LEFT LED")
    print("  d/D - RIGHT LED")
    print("  w/W - UP LED")
    print("  s/S - DOWN LED")
    print("  space - ALL LEDs")
    print("  q/Q - Quit")
    print("\nPress any key...")
    
    # Tắt tất cả LED
    for pin in LED_PINS.values():
        GPIO.output(pin, GPIO.LOW)
    
    try:
        import tty
        import termios
        
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        
        while True:
            char = sys.stdin.read(1)
            
            # Tắt tất cả LED trước
            for pin in LED_PINS.values():
                GPIO.output(pin, GPIO.LOW)
            
            if char.lower() == 'q':
                break
            elif char.lower() == 'a':
                GPIO.output(LED_PINS['left'], GPIO.HIGH)
                print("\r← LEFT  ", end='', flush=True)
            elif char.lower() == 'd':
                GPIO.output(LED_PINS['right'], GPIO.HIGH)
                print("\r→ RIGHT ", end='', flush=True)
            elif char.lower() == 'w':
                GPIO.output(LED_PINS['up'], GPIO.HIGH)
                print("\r↑ UP    ", end='', flush=True)
            elif char.lower() == 's':
                GPIO.output(LED_PINS['down'], GPIO.HIGH)
                print("\r↓ DOWN  ", end='', flush=True)
            elif char == ' ':
                for pin in LED_PINS.values():
                    GPIO.output(pin, GPIO.HIGH)
                print("\r✦ ALL   ", end='', flush=True)
        
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
    except ImportError:
        print("\nInteractive mode not available (tty/termios not found)")
    except Exception as e:
        print(f"\nError in interactive mode: {e}")

def main():
    """Main function"""
    print("\n" + "="*50)
    print("  LED Connection Test for Raspberry Pi")
    print("="*50)
    print("\nGPIO Pin Mapping:")
    for direction, pin in LED_PINS.items():
        print(f"  {direction.upper():>6}: GPIO {pin:>2}")
    
    try:
        setup_gpio()
        
        # Các test
        test_individual_leds()
        time.sleep(1)
        
        test_all_leds_together()
        time.sleep(1)
        
        test_pattern()
        time.sleep(1)
        
        test_blink()
        time.sleep(1)
        
        # Interactive test (optional)
        try:
            interactive_test()
        except KeyboardInterrupt:
            pass
        
        print("\n\n" + "="*50)
        print("  All tests completed successfully! ✓")
        print("="*50)
        print("\nIf all LEDs lit up correctly, your wiring is good!")
        print("You can now run the gaze estimation program:\n")
        print("  python3 gaze_led_rpi.py --model weights/mobileone_s0_gaze.onnx\n")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    
    except Exception as e:
        print(f"\n\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Check LED polarity (long leg = +, short leg = -)")
        print("2. Check resistor connections")
        print("3. Verify GPIO pin numbers")
        print("4. Run with sudo: sudo python3 test_led.py")
    
    finally:
        cleanup_gpio()

if __name__ == '__main__':
    main()
