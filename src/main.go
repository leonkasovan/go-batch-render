// main.go - Fixed GIF color and vertical flip
package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/gif"
	"image/png"
	"log"
	"os"
	"runtime"
	"strings"
	"time"

	"go-batch-render/packages/gl/v3.3-core/gl"
	"go-batch-render/packages/glfw"
	"github.com/golang/freetype/truetype"
	"golang.org/x/image/font"
	"golang.org/x/image/math/fixed"
)

// ---- config ----
var (
	windowWidth  = 640
	windowHeight = 480

	atlasWidth  = 1024
	atlasHeight = 1024

	fontPath    string
	gifPath     string
	fontDPI     = 72
	fontPixelEm = 32
	padding     = 2

	useVSync = true
)

// FPS counter and other types remain the same...
type FPSCounter struct {
	frames   int
	lastTime time.Time
	fps      float64
}

func NewFPSCounter() *FPSCounter {
	return &FPSCounter{
		frames:   0,
		lastTime: time.Now(),
		fps:      0,
	}
}

func (f *FPSCounter) Update() {
	f.frames++
	if time.Since(f.lastTime) >= time.Second {
		f.fps = float64(f.frames) / time.Since(f.lastTime).Seconds()
		f.frames = 0
		f.lastTime = time.Now()
	}
}

func (f *FPSCounter) GetFPS() float64 {
	return f.fps
}

type Sprite struct {
	ID       string
	W, H     int
	AtlasX   int
	AtlasY   int
	Advance  int
	BearingX int
	BearingY int
	U1, V1   float32
	U2, V2   float32
}

type SpriteAtlasType int

const (
	AtlasTypeFont SpriteAtlasType = iota
	AtlasTypeGIF
)

type SpriteAtlasConfig struct {
	Type     SpriteAtlasType
	FontPath string
	GIFPath  string
	FontSize int
	Padding  int
	AtlasW   int
	AtlasH   int
}

type SpriteAtlasResult struct {
	Atlas   *GLAtlas
	Sprites map[string]*Sprite
	Type    SpriteAtlasType
}

type TextCommand struct {
	text     string
	x, y     int
	color    [3]float32
	isSprite bool
}

type BatchRenderer struct {
	prog     uint32
	vao      uint32
	vbo      uint32
	sprites  map[string]*Sprite
	atlas    *GLAtlas
	vertices []float32
	commands []TextCommand
}

func NewBatchRenderer(prog, vao, vbo uint32, sprites map[string]*Sprite, atlas *GLAtlas) *BatchRenderer {
	return &BatchRenderer{
		prog:     prog,
		vao:      vao,
		vbo:      vbo,
		sprites:  sprites,
		atlas:    atlas,
		vertices: make([]float32, 0, 1024),
		commands: make([]TextCommand, 0, 64),
	}
}

func (b *BatchRenderer) Draw(text string, x, y int, color [3]float32) {
	b.commands = append(b.commands, TextCommand{
		text:  text,
		x:     x,
		y:     y,
		color: color,
	})
}

func (b *BatchRenderer) DrawSprite(spriteID string, x, y int, color [3]float32) {
	b.commands = append(b.commands, TextCommand{
		text:     spriteID,
		x:        x,
		y:        y,
		color:    color,
		isSprite: true,
	})
}

func (b *BatchRenderer) Flush() {
	if len(b.commands) == 0 {
		return
	}

	b.vertices = b.vertices[:0]

	for _, cmd := range b.commands {
		if cmd.isSprite {
			b.appendSpriteVertices(cmd.text, cmd.x, cmd.y, cmd.color)
		} else {
			b.appendTextVertices(cmd.text, cmd.x, cmd.y, cmd.color)
		}
	}

	if len(b.vertices) > 0 {
		gl.BindVertexArray(b.vao)
		gl.BindBuffer(gl.ARRAY_BUFFER, b.vbo)
		gl.BufferData(gl.ARRAY_BUFFER, len(b.vertices)*4, gl.Ptr(b.vertices), gl.DYNAMIC_DRAW)
		cnt := int32(len(b.vertices) / 7)
		gl.DrawArrays(gl.TRIANGLES, 0, cnt)
		gl.BindVertexArray(0)
	}

	b.commands = b.commands[:0]
}

func (b *BatchRenderer) GetVertexCount() int {
	return len(b.vertices)
}

func (b *BatchRenderer) GetCommandCount() int {
	return len(b.commands)
}

func (b *BatchRenderer) appendTextVertices(s string, startX, startY int, textColor [3]float32) {
	x := float32(startX)
	y := float32(startY)

	for _, ch := range s {
		spriteID := string(ch)
		sprite, ok := b.sprites[spriteID]
		if !ok {
			if space, ok := b.sprites[" "]; ok {
				x += float32(space.Advance)
			} else {
				x += 8
			}
			continue
		}

		b.appendSingleSpriteVertices(sprite, x, y, textColor)
		x += float32(sprite.Advance)
	}
}

func (b *BatchRenderer) appendSpriteVertices(spriteID string, startX, startY int, spriteColor [3]float32) {
	sprite, ok := b.sprites[spriteID]
	if !ok {
		return
	}
	b.appendSingleSpriteVertices(sprite, float32(startX), float32(startY), spriteColor)
}

func (b *BatchRenderer) appendSingleSpriteVertices(sprite *Sprite, x, y float32, spriteColor [3]float32) {
	// Fixed: No vertical flip - use proper Y coordinates
	x0 := x + float32(sprite.BearingX)
	y0 := y + float32(sprite.BearingY) // Fixed: Remove the flip adjustment
	x1 := x0 + float32(sprite.W)
	y1 := y0 + float32(sprite.H) // Fixed: No subtraction

	u1 := sprite.U1
	v1 := sprite.V1
	u2 := sprite.U2
	v2 := sprite.V2

	b.vertices = append(b.vertices,
		x0, y0, u1, v1, spriteColor[0], spriteColor[1], spriteColor[2],
		x1, y0, u2, v1, spriteColor[0], spriteColor[1], spriteColor[2],
		x1, y1, u2, v2, spriteColor[0], spriteColor[1], spriteColor[2],
		x0, y0, u1, v1, spriteColor[0], spriteColor[1], spriteColor[2],
		x1, y1, u2, v2, spriteColor[0], spriteColor[1], spriteColor[2],
		x0, y1, u1, v2, spriteColor[0], spriteColor[1], spriteColor[2],
	)
}

type SimpleAtlas struct {
	W, H int
	x, y int
	rowH int
}

func NewSimpleAtlas(w, h int) *SimpleAtlas {
	return &SimpleAtlas{
		W:    w,
		H:    h,
		x:    0,
		y:    0,
		rowH: 0,
	}
}

func (a *SimpleAtlas) Pack(wid, hei int) (int, int) {
	if a.x+wid > a.W {
		a.x = 0
		a.y += a.rowH
		a.rowH = 0
	}

	if a.y+hei > a.H {
		return -1, -1
	}

	if hei > a.rowH {
		a.rowH = hei
	}

	x, y := a.x, a.y
	a.x += wid
	return x, y
}

type GLAtlas struct {
	Tex  uint32
	W, H int
	CPU  []uint8
}

func NewGLAtlas(w, h int) *GLAtlas {
	var tex uint32
	gl.GenTextures(1, &tex)
	gl.BindTexture(gl.TEXTURE_2D, tex)
	gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGBA, int32(w), int32(h), 0, gl.RGBA, gl.UNSIGNED_BYTE, nil)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
	return &GLAtlas{Tex: tex, W: w, H: h, CPU: make([]uint8, w*h*4)}
}

func (a *GLAtlas) UploadSubImage(x, y, w, h int, pixels []uint8) {
	if len(pixels) < w*h*4 {
		log.Printf("UploadSubImage: pixels slice too small. Expected %d, got %d", w*h*4, len(pixels))
		return
	}
	for row := 0; row < h; row++ {
		dstOff := ((y+row)*a.W + x) * 4
		srcOff := row * w * 4
		copy(a.CPU[dstOff:dstOff+w*4], pixels[srcOff:srcOff+w*4])
	}
	gl.BindTexture(gl.TEXTURE_2D, a.Tex)
	gl.PixelStorei(gl.UNPACK_ALIGNMENT, 1)
	gl.TexSubImage2D(gl.TEXTURE_2D, 0, int32(x), int32(y), int32(w), int32(h), gl.RGBA, gl.UNSIGNED_BYTE, gl.Ptr(&pixels[0]))
}

func (a *GLAtlas) DumpPNG(fname string) error {
	img := image.NewRGBA(image.Rect(0, 0, a.W, a.H))
	copy(img.Pix, a.CPU)
	f, err := os.Create(fname)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

// RasterizeGlyphFace - updated to use RGBA
func RasterizeGlyphFace(face font.Face, r rune, pad int) ([]uint8, int, int, int, int, int) {
	metrics := face.Metrics()
	ascent := metrics.Ascent.Ceil()
	height := ascent + metrics.Descent.Ceil()

	advance, _ := face.GlyphAdvance(r)
	adv := (int(advance) + 32) / 64
	if adv < 1 {
		adv = 8
	}

	width := adv
	if width < 8 {
		width = 8
	}

	W := width + pad*2
	H := height + pad*2

	img := image.NewRGBA(image.Rect(0, 0, W, H))
	draw.Draw(img, img.Bounds(), &image.Uniform{color.RGBA{0, 0, 0, 0}}, image.Point{}, draw.Src)

	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(color.White),
		Face: face,
		Dot: fixed.Point26_6{
			X: fixed.I(pad),
			Y: fixed.I(pad + ascent),
		},
	}
	d.DrawString(string(r))

	pix := make([]uint8, W*H*4)
	copy(pix, img.Pix)

	bearingX := 0
	bearingY := ascent

	return pix, W, H, adv, bearingX, bearingY
}

// CreateFontAtlas
func CreateFontAtlas(config SpriteAtlasConfig) (*SpriteAtlasResult, error) {
	fontBytes, err := os.ReadFile(config.FontPath)
	if err != nil {
		return nil, fmt.Errorf("read font %s: %v", config.FontPath, err)
	}
	tt, err := truetype.Parse(fontBytes)
	if err != nil {
		return nil, fmt.Errorf("parse ttf: %v", err)
	}
	opts := &truetype.Options{Size: float64(config.FontSize), DPI: float64(fontDPI), Hinting: font.HintingFull}
	face := truetype.NewFace(tt, opts)
	defer face.Close()

	atlas := NewSimpleAtlas(config.AtlasW, config.AtlasH)
	glAtlas := NewGLAtlas(config.AtlasW, config.AtlasH)
	sprites := make(map[string]*Sprite)

	runes := []rune{}
	for r := rune(32); r <= 126; r++ {
		runes = append(runes, r)
	}

	log.Println("Packing font glyphs into atlas...")
	for _, r := range runes {
		pix, w, h, adv, bx, by := RasterizeGlyphFace(face, r, config.Padding)
		x, y := atlas.Pack(w, h)
		if x == -1 {
			continue
		}
		glAtlas.UploadSubImage(x, y, w, h, pix)
		sprite := &Sprite{
			ID:       string(r),
			W:        w,
			H:        h,
			AtlasX:   x,
			AtlasY:   y,
			Advance:  adv,
			BearingX: bx,
			BearingY: by,
		}
		sprite.U1 = float32(x) / float32(config.AtlasW)
		sprite.V1 = float32(y) / float32(config.AtlasH)
		sprite.U2 = float32(x+w) / float32(config.AtlasW)
		sprite.V2 = float32(y+h) / float32(config.AtlasH)
		sprites[string(r)] = sprite
	}
	log.Printf("Successfully packed %d font glyphs into atlas", len(sprites))

	return &SpriteAtlasResult{
		Atlas:   glAtlas,
		Sprites: sprites,
		Type:    AtlasTypeFont,
	}, nil
}

// imageToRGBAWithPalette converts GIF image to RGBA using its palette
func imageToRGBAWithPalette(img image.Image, palette color.Palette) *image.RGBA {
	bounds := img.Bounds()
	rgba := image.NewRGBA(bounds)

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			// Get the color from the image using its palette
			c := img.At(x, y)
			if palette != nil {
				// Convert palette color to RGBA
				rgba.Set(x, y, c)
			} else {
				// Fallback: use the color directly
				rgba.Set(x, y, c)
			}
		}
	}
	return rgba
}

// CreateGIFAtlas - fixed to handle GIF palette and vertical orientation
func CreateGIFAtlas(config SpriteAtlasConfig) (*SpriteAtlasResult, error) {
	file, err := os.Open(config.GIFPath)
	if err != nil {
		return nil, fmt.Errorf("open GIF %s: %v", config.GIFPath, err)
	}
	defer file.Close()

	gifData, err := gif.DecodeAll(file)
	if err != nil {
		return nil, fmt.Errorf("decode GIF %s: %v", config.GIFPath, err)
	}

	if len(gifData.Image) == 0 {
		return nil, fmt.Errorf("GIF contains no frames")
	}

	atlas := NewSimpleAtlas(config.AtlasW, config.AtlasH)
	glAtlas := NewGLAtlas(config.AtlasW, config.AtlasH)
	sprites := make(map[string]*Sprite)

	log.Printf("Packing %d GIF frames into atlas...", len(gifData.Image))

	for i, img := range gifData.Image {
		// Use the GIF's palette to convert to RGBA with correct colors
		var rgba *image.RGBA
		if gifData.Image[i].Palette != nil {
			rgba = imageToRGBAWithPalette(img, gifData.Image[i].Palette)
		} else {
			rgba = imageToRGBAWithPalette(img, nil)
		}

		bounds := rgba.Bounds()
		w := bounds.Dx()
		h := bounds.Dy()

		// Create pixel data - no vertical flip needed
		pixels := make([]uint8, w*h*4)

		// Copy pixels row by row in correct order (no flip)
		for y := 0; y < h; y++ {
			srcOffset := y * rgba.Stride
			dstOffset := y * w * 4
			copy(pixels[dstOffset:dstOffset+w*4], rgba.Pix[srcOffset:srcOffset+w*4])
		}

		x, y := atlas.Pack(w, h)
		if x == -1 {
			log.Printf("atlas full, cannot pack GIF frame %d, skipping", i)
			continue
		}

		glAtlas.UploadSubImage(x, y, w, h, pixels)

		sprite := &Sprite{
			ID:       fmt.Sprintf("frame_%d", i),
			W:        w,
			H:        h,
			AtlasX:   x,
			AtlasY:   y,
			Advance:  w,
			BearingX: 0,
			BearingY: 0, // No bearing for sprites
		}
		sprite.U1 = float32(x) / float32(config.AtlasW)
		sprite.V1 = float32(y) / float32(config.AtlasH)
		sprite.U2 = float32(x+w) / float32(config.AtlasW)
		sprite.V2 = float32(y+h) / float32(config.AtlasH)
		sprites[sprite.ID] = sprite

		log.Printf("Packed GIF frame %d at (%d, %d) size %dx%d", i, x, y, w, h)
	}

	if len(sprites) == 0 {
		return nil, fmt.Errorf("no GIF frames were successfully packed into atlas")
	}

	log.Printf("Successfully packed %d GIF frames into atlas", len(sprites))

	return &SpriteAtlasResult{
		Atlas:   glAtlas,
		Sprites: sprites,
		Type:    AtlasTypeGIF,
	}, nil
}

// CreateSpriteAtlas
func CreateSpriteAtlas(config SpriteAtlasConfig) (*SpriteAtlasResult, error) {
	switch config.Type {
	case AtlasTypeFont:
		return CreateFontAtlas(config)
	case AtlasTypeGIF:
		return CreateGIFAtlas(config)
	default:
		return nil, fmt.Errorf("unknown atlas type: %v", config.Type)
	}
}

// Updated shader to handle full color textures
const vertexShader = `#version 330 core
layout(location=0) in vec2 inPos;
layout(location=1) in vec2 inUV;
layout(location=2) in vec3 inColor;
out vec2 vUV;
out vec3 vColor;
uniform mat4 uProj;
void main() {
    vUV = inUV;
    vColor = inColor;
    gl_Position = uProj * vec4(inPos,0.0,1.0);
}` + "\x00"

const fragmentShader = `#version 330 core
in vec2 vUV;
in vec3 vColor;
out vec4 FragColor;
uniform sampler2D uTex;
void main() {
    vec4 texColor = texture(uTex, vUV);
    // Use texture RGB and multiply by vertex color, use texture alpha
    FragColor = vec4(texColor.rgb * vColor, texColor.a);
}` + "\x00"

// GL helpers remain the same...
func init() {
	runtime.LockOSThread()
}

func initGLFW() {
	if err := glfw.Init(); err != nil {
		log.Fatalln("glfw init:", err)
	}
}

func initGL() {
	if err := gl.Init(); err != nil {
		log.Fatalln("gl init:", err)
	}
	log.Println("GL version:", gl.GoStr(gl.GetString(gl.VERSION)))
}

func compileShader(src string, shaderType uint32) uint32 {
	sh := gl.CreateShader(shaderType)
	cstrs, free := gl.Strs(src)
	gl.ShaderSource(sh, 1, cstrs, nil)
	free()
	gl.CompileShader(sh)
	var status int32
	gl.GetShaderiv(sh, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		var l int32
		gl.GetShaderiv(sh, gl.INFO_LOG_LENGTH, &l)
		logstr := strings.Repeat("\x00", int(l+1))
		gl.GetShaderInfoLog(sh, l, nil, gl.Str(logstr))
		log.Fatalln("shader compile error:", logstr)
	}
	return sh
}

func linkProgram(vs, fs uint32) uint32 {
	p := gl.CreateProgram()
	gl.AttachShader(p, vs)
	gl.AttachShader(p, fs)
	gl.LinkProgram(p)
	var status int32
	gl.GetProgramiv(p, gl.LINK_STATUS, &status)
	if status == gl.FALSE {
		var l int32
		gl.GetProgramiv(p, gl.INFO_LOG_LENGTH, &l)
		logstr := strings.Repeat("\x00", int(l+1))
		gl.GetProgramInfoLog(p, l, nil, gl.Str(logstr))
		log.Fatalln("link error:", logstr)
	}
	return p
}

func ortho(left, right, bottom, top, near, far float32) [16]float32 {
	rl := right - left
	tb := top - bottom
	fn := far - near
	return [16]float32{
		2.0 / rl, 0, 0, 0,
		0, 2.0 / tb, 0, 0,
		0, 0, -2.0 / fn, 0,
		-(right + left) / rl, -(top + bottom) / tb, -(far + near) / fn, 1,
	}
}

// ---- main ----
func main() {
	flag.StringVar(&fontPath, "font", "", "Path to TTF font file")
	flag.StringVar(&gifPath, "gif", "", "Path to GIF animation file")
	flag.IntVar(&windowWidth, "width", 800, "Window width")
	flag.IntVar(&windowHeight, "height", 600, "Window height")
	flag.IntVar(&fontPixelEm, "size", 32, "Font size in pixels")
	flag.BoolVar(&useVSync, "vsync", true, "Enable VSync")
	flag.Parse()

	if fontPath == "" && gifPath == "" {
		log.Fatal("Please specify either a font file using -font or a GIF file using -gif")
	}

	if fontPath != "" {
		if _, err := os.Stat(fontPath); os.IsNotExist(err) {
			log.Fatalf("Font file not found: %s", fontPath)
		}
	}
	if gifPath != "" {
		if _, err := os.Stat(gifPath); os.IsNotExist(err) {
			log.Fatalf("GIF file not found: %s", gifPath)
		}
	}

	log.Printf("Using font: %s, GIF: %s (size: %d, VSync: %v)", fontPath, gifPath, fontPixelEm, useVSync)

	initGLFW()
	defer glfw.Terminate()

	glfw.WindowHint(glfw.ContextVersionMajor, 3)
	glfw.WindowHint(glfw.ContextVersionMinor, 3)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.OpenGLForwardCompatible, glfw.True)

	win, err := glfw.CreateWindow(windowWidth, windowHeight, "Sprite Renderer (Batch)", nil, nil)
	if err != nil {
		log.Fatalln("create window:", err)
	}
	win.MakeContextCurrent()

	if useVSync {
		glfw.SwapInterval(1)
		log.Println("VSync enabled")
	} else {
		glfw.SwapInterval(0)
		log.Println("VSync disabled")
	}

	initGL()

	var atlasResult *SpriteAtlasResult

	if gifPath != "" {
		config := SpriteAtlasConfig{
			Type:    AtlasTypeGIF,
			GIFPath: gifPath,
			Padding: padding,
			AtlasW:  atlasWidth,
			AtlasH:  atlasHeight,
		}
		atlasResult, err = CreateSpriteAtlas(config)
		if err != nil {
			log.Fatalf("Failed to create GIF atlas: %v", err)
		}
		log.Println("Loaded GIF animation atlas")
	} else {
		config := SpriteAtlasConfig{
			Type:     AtlasTypeFont,
			FontPath: fontPath,
			FontSize: fontPixelEm,
			Padding:  padding,
			AtlasW:   atlasWidth,
			AtlasH:   atlasHeight,
		}
		atlasResult, err = CreateSpriteAtlas(config)
		if err != nil {
			log.Fatalf("Failed to create font atlas: %v", err)
		}
		log.Println("Loaded font atlas")
	}

	if err := atlasResult.Atlas.DumpPNG("atlas_debug.png"); err != nil {
		log.Println("dump atlas png failed:", err)
	} else {
		log.Println("wrote atlas_debug.png")
	}

	vs := compileShader(vertexShader, gl.VERTEX_SHADER)
	fs := compileShader(fragmentShader, gl.FRAGMENT_SHADER)
	prog := linkProgram(vs, fs)
	gl.DeleteShader(vs)
	gl.DeleteShader(fs)

	var vao, vbo uint32
	gl.GenVertexArrays(1, &vao)
	gl.GenBuffers(1, &vbo)
	gl.BindVertexArray(vao)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)

	initialBufferSize := 65536 * 2
	gl.BufferData(gl.ARRAY_BUFFER, initialBufferSize, nil, gl.DYNAMIC_DRAW)

	stride := int32(7 * 4)
	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(0, 2, gl.FLOAT, false, stride, gl.PtrOffset(0))
	gl.EnableVertexAttribArray(1)
	gl.VertexAttribPointer(1, 2, gl.FLOAT, false, stride, gl.PtrOffset(2*4))
	gl.EnableVertexAttribArray(2)
	gl.VertexAttribPointer(2, 3, gl.FLOAT, false, stride, gl.PtrOffset(4*4))
	gl.BindVertexArray(0)

	gl.UseProgram(prog)
	projLoc := gl.GetUniformLocation(prog, gl.Str("uProj\x00"))
	proj := ortho(0, float32(windowWidth), float32(windowHeight), 0, -1, 1)
	gl.UniformMatrix4fv(projLoc, 1, false, &proj[0])

	texLoc := gl.GetUniformLocation(prog, gl.Str("uTex\x00"))
	gl.Uniform1i(texLoc, 0)

	gl.Enable(gl.BLEND)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)

	batchRenderer := NewBatchRenderer(prog, vao, vbo, atlasResult.Sprites, atlasResult.Atlas)
	fpsCounter := NewFPSCounter()

	// startTime := time.Now()
	frameCount := 0
	gifFrameIndex := 0
	lastFrameTime := time.Now()

	log.Println("Starting batch rendering loop...")

	for !win.ShouldClose() {
		glfw.PollEvents()
		gl.ClearColor(0.1, 0.1, 0.1, 1.0)
		gl.Clear(gl.COLOR_BUFFER_BIT)

		fpsCounter.Update()
		fps := fpsCounter.GetFPS()
		// elapsed := time.Since(startTime).Seconds()

		gl.UseProgram(prog)
		gl.ActiveTexture(gl.TEXTURE0)
		gl.BindTexture(gl.TEXTURE_2D, atlasResult.Atlas.Tex)

		if atlasResult.Type == AtlasTypeFont {
			// waveOffset := math.Sin(elapsed*2) * 50
			// pulse := 0.5 + 0.5*math.Sin(elapsed*3)

			batchRenderer.Draw("Batch Font Renderer Demo", 50, 50, [3]float32{1.0, 0.5, 0.2})
			batchRenderer.Draw(fmt.Sprintf("FPS: %.1f (VSync: %v)", fps, useVSync), 50, 90, [3]float32{0.2, 0.8, 1.0})
			batchRenderer.Draw(fmt.Sprintf("Frame: %d", frameCount), 50, 120, [3]float32{0.7, 0.7, 0.7})

		} else {
			batchRenderer.Draw("GIF Animation Renderer", 50, 50, [3]float32{1.0, 0.5, 0.2})
			batchRenderer.Draw(fmt.Sprintf("FPS: %.1f (VSync: %v)", fps, useVSync), 50, 90, [3]float32{0.2, 0.8, 1.0})
			batchRenderer.Draw(fmt.Sprintf("Frame: %d", frameCount), 50, 120, [3]float32{0.7, 0.7, 0.7})
			batchRenderer.Draw(fmt.Sprintf("GIF Frame: %d/%d", gifFrameIndex, len(atlasResult.Sprites)), 50, 150, [3]float32{0.7, 0.7, 0.7})

			if time.Since(lastFrameTime) > time.Millisecond*100 {
				gifFrameIndex = (gifFrameIndex + 1) % len(atlasResult.Sprites)
				lastFrameTime = time.Now()
			}

			frameID := fmt.Sprintf("frame_%d", gifFrameIndex)
			if _, exists := atlasResult.Sprites[frameID]; exists {
				// Use white color to preserve original GIF colors
				batchRenderer.DrawSprite(frameID, 200, 200, [3]float32{1.0, 1.0, 1.0})
			}

			// Draw all frames for debugging
			for i := 0; i < len(atlasResult.Sprites); i++ {
				frameID := fmt.Sprintf("frame_%d", i)
				if sprite, exists := atlasResult.Sprites[frameID]; exists {
					x := 50 + (i%4)*(sprite.W+10)
					y := 300 + (i/4)*(sprite.H+10)
					batchRenderer.DrawSprite(frameID, x, y, [3]float32{1.0, 1.0, 1.0})
				}
			}
		}

		batchRenderer.Flush()
		frameCount++
		win.SwapBuffers()
	}
}
