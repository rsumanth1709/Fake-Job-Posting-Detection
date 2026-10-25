import os
import math
from PIL import Image, ImageDraw

def render_highres_icon(size=512):
    # Render at 4x supersampling (2048x2048) for ultra-sharp antialiasing
    scale = 4
    canvas_size = size * scale
    img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    def sc(coord):
        # Maps 0-64 viewBox to canvas_size
        return coord * (canvas_size / 64.0)

    # 1. Subtle drop shadow
    shadow_offset = sc(2.5)
    shadow_polygon = [
        (sc(32), sc(4) + shadow_offset),
        (sc(54), sc(12) + shadow_offset),
        (sc(54), sc(36) + shadow_offset),
        (sc(43), sc(51) + shadow_offset),
        (sc(32), sc(60) + shadow_offset),
        (sc(21), sc(51) + shadow_offset),
        (sc(10), sc(36) + shadow_offset),
        (sc(10), sc(12) + shadow_offset),
    ]
    draw.polygon(shadow_polygon, fill=(15, 23, 42, 100))

    # 2. Outer Shield Body with gradient
    # We create vertical slices or fill polygon with rich cyber blue
    shield_polygon = [
        (sc(32), sc(4)),
        (sc(54), sc(12)),
        (sc(54), sc(36)),
        (sc(43), sc(51)),
        (sc(32), sc(60)),
        (sc(21), sc(51)),
        (sc(10), sc(36)),
        (sc(10), sc(12)),
    ]
    
    # Gradient simulation
    mask = Image.new("L", (canvas_size, canvas_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.polygon(shield_polygon, fill=255)
    
    gradient = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    for y in range(canvas_size):
        ratio = y / float(canvas_size)
        r = int(30 * (1 - ratio) + 2 * ratio)
        g = int(58 * (1 - ratio) + 132 * ratio)
        b = int(186 * (1 - ratio) + 235 * ratio)
        line_draw = ImageDraw.Draw(gradient)
        line_draw.line([(0, y), (canvas_size, y)], fill=(r, g, b, 255))
    
    img.paste(gradient, (0, 0), mask)

    # Outer Shield Stroke
    draw.polygon(shield_polygon, outline=(96, 165, 250, 255), width=int(sc(1.5)))

    # 3. Inner Tech Accent Border
    inner_polygon = [
        (sc(32), sc(9)),
        (sc(49), sc(15)),
        (sc(49), sc(34)),
        (sc(40), sc(46)),
        (sc(32), sc(54)),
        (sc(24), sc(46)),
        (sc(15), sc(34)),
        (sc(15), sc(15)),
    ]
    draw.polygon(inner_polygon, outline=(147, 197, 253, 130), width=int(sc(1.0)))

    # 4. Circuit lines & nodes
    # Central stem
    draw.line([(sc(32), sc(17)), (sc(32), sc(25))], fill=(56, 189, 248, 255), width=int(sc(2.0)))
    draw.ellipse([sc(32-2.5), sc(17-2.5), sc(32+2.5), sc(17+2.5)], fill=(56, 189, 248, 255))

    # Left branch
    draw.line([(sc(22), sc(25)), (sc(28), sc(27))], fill=(56, 189, 248, 200), width=int(sc(1.5)))
    draw.ellipse([sc(22-2), sc(25-2), sc(22+2), sc(25+2)], fill=(96, 165, 250, 255))

    # Right branch
    draw.line([(sc(42), sc(25)), (sc(36), sc(27))], fill=(56, 189, 248, 200), width=int(sc(1.5)))
    draw.ellipse([sc(42-2), sc(25-2), sc(42+2), sc(25+2)], fill=(96, 165, 250, 255))

    # 5. Bold Verification Checkmark (Vibrant Emerald/Cyan Glow)
    check_pts = [
        (sc(22), sc(34)),
        (sc(29), sc(41)),
        (sc(43), sc(25))
    ]
    # Draw thicker rounded checkmark
    draw.line(check_pts, fill=(16, 185, 129, 255), width=int(sc(4.5)), joint="curve")
    
    # Glowing tip
    draw.ellipse([sc(43-2), sc(25-2), sc(43+2), sc(25+2)], fill=(255, 255, 255, 255))

    # Supersampled resize to target size with LANCZOS filter
    final_img = img.resize((size, size), Image.Resampling.LANCZOS)
    return final_img

def main():
    os.makedirs("static", exist_ok=True)
    
    # 512x512
    img512 = render_highres_icon(512)
    img512.save("static/icon-512x512.png", "PNG")
    print("Saved static/icon-512x512.png")

    # 180x180 (Apple touch icon)
    img180 = img512.resize((180, 180), Image.Resampling.LANCZOS)
    img180.save("static/apple-touch-icon.png", "PNG")
    print("Saved static/apple-touch-icon.png")

    # 32x32
    img32 = img512.resize((32, 32), Image.Resampling.LANCZOS)
    img32.save("static/favicon-32x32.png", "PNG")
    print("Saved static/favicon-32x32.png")

    # 16x16
    img16 = img512.resize((16, 16), Image.Resampling.LANCZOS)
    img16.save("static/favicon-16x16.png", "PNG")
    print("Saved static/favicon-16x16.png")

    # Multi-resolution ICO: 16, 32, 48, 64
    img48 = img512.resize((48, 48), Image.Resampling.LANCZOS)
    img64 = img512.resize((64, 64), Image.Resampling.LANCZOS)
    
    img32.save(
        "static/favicon.ico", 
        format="ICO", 
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64)],
        append_images=[img16, img48, img64]
    )
    print("Saved static/favicon.ico")

if __name__ == "__main__":
    main()
