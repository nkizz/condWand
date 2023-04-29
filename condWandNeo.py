import board
import neopixel

print("Starting")
pixels = neopixel.NeoPixel(board.NEOPIXEL, 10, brightness=1, auto_write=False)
pixels.fill((0, 0, 0))
pixels.show()
while True:
    #read a RGB color value from the usb serial and print it
    rgb = input().split(',')
    #check if rgb is a valid color
    if len(rgb) == 3:
        try:
            r = int(rgb[0])
            g = int(rgb[1])
            b = int(rgb[2])
            if r >= 0 and r <= 255 and g >= 0 and g <= 255 and b >= 0 and b <= 255:
                pixels.fill((r, g, b))
                pixels.show()
        except:
            pass