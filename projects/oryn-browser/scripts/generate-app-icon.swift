import AppKit
import Foundation

let arguments = CommandLine.arguments
guard arguments.count == 2 else {
  FileHandle.standardError.write(Data("Usage: generate-app-icon.swift <output.icns>\n".utf8))
  exit(2)
}

let output = URL(fileURLWithPath: arguments[1]).standardizedFileURL
let iconset = FileManager.default.temporaryDirectory
  .appendingPathComponent("OrynBrowser-\(UUID().uuidString).iconset", isDirectory: true)
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
defer { try? FileManager.default.removeItem(at: iconset) }

let variants: [(String, Int)] = [
  ("icon_16x16.png", 16),
  ("icon_16x16@2x.png", 32),
  ("icon_32x32.png", 32),
  ("icon_32x32@2x.png", 64),
  ("icon_128x128.png", 128),
  ("icon_128x128@2x.png", 256),
  ("icon_256x256.png", 256),
  ("icon_256x256@2x.png", 512),
  ("icon_512x512.png", 512),
  ("icon_512x512@2x.png", 1024),
]

for (name, size) in variants {
  let representation = NSBitmapImageRep(
    bitmapDataPlanes: nil,
    pixelsWide: size,
    pixelsHigh: size,
    bitsPerSample: 8,
    samplesPerPixel: 4,
    hasAlpha: true,
    isPlanar: false,
    colorSpaceName: .deviceRGB,
    bytesPerRow: 0,
    bitsPerPixel: 0
  )!
  NSGraphicsContext.saveGraphicsState()
  NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: representation)
  let dimension = CGFloat(size)
  let inset = dimension * 0.06
  let bounds = NSRect(x: inset, y: inset, width: dimension - inset * 2, height: dimension - inset * 2)
  let background = NSBezierPath(roundedRect: bounds, xRadius: dimension * 0.22, yRadius: dimension * 0.22)
  NSGradient(
    colors: [
      NSColor(red: 0.36, green: 0.36, blue: 0.89, alpha: 1),
      NSColor(red: 0.22, green: 0.22, blue: 0.60, alpha: 1),
    ]
  )!.draw(in: background, angle: -55)

  let ringInset = dimension * 0.27
  let ring = NSBezierPath(ovalIn: NSRect(
    x: ringInset,
    y: ringInset,
    width: dimension - ringInset * 2,
    height: dimension - ringInset * 2
  ))
  ring.lineWidth = max(1.5, dimension * 0.105)
  ring.lineCapStyle = .round
  NSColor.white.setStroke()
  ring.stroke()

  let dotSize = max(2.0, dimension * 0.09)
  let dot = NSBezierPath(ovalIn: NSRect(
    x: dimension * 0.68,
    y: dimension * 0.68,
    width: dotSize,
    height: dotSize
  ))
  NSColor(red: 0.95, green: 0.95, blue: 1.0, alpha: 0.92).setFill()
  dot.fill()
  NSGraphicsContext.restoreGraphicsState()

  let data = representation.representation(using: .png, properties: [:])!
  try data.write(to: iconset.appendingPathComponent(name), options: .atomic)
}

try FileManager.default.createDirectory(
  at: output.deletingLastPathComponent(),
  withIntermediateDirectories: true
)
let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
process.arguments = ["--convert", "icns", "--output", output.path, iconset.path]
try process.run()
process.waitUntilExit()
guard process.terminationStatus == 0 else {
  throw NSError(domain: "OrynIcon", code: Int(process.terminationStatus))
}
