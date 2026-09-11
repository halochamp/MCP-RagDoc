import Foundation
import Vision
import AppKit

let args = CommandLine.arguments

// --langs : print supported recognition languages (accurate level)
if args.count > 1 && args[1] == "--langs" {
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    if let langs = try? req.supportedRecognitionLanguages() {
        print(langs.joined(separator: ", "))
    }
    exit(0)
}

guard args.count > 1,
      let img = NSImage(contentsOfFile: args[1]),
      let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    print("ERR: cannot load \(args.count > 1 ? args[1] : "<no path>")")
    exit(1)
}

// Output: one observation per line as
// x\ty\tw\th\tconfidence\tcandidate1\tcandidate2\tcandidate3
// (boundingBox is normalized [0,1], origin BOTTOM-LEFT). Candidate text is
// flattened to one TSV-safe line; Python keeps backward compatibility with the
// older 5-field helper protocol.
func tsvSafe(_ text: String) -> String {
    return text
        .replacingOccurrences(of: "\t", with: " ")
        .replacingOccurrences(of: "\r", with: " ")
        .replacingOccurrences(of: "\n", with: " ")
}

let req = VNRecognizeTextRequest { (request, _) in
    guard let obs = request.results as? [VNRecognizedTextObservation] else { return }
    var out: [String] = []
    for o in obs {
        let candidates = o.topCandidates(3)
        guard let best = candidates.first else { continue }
        let b = o.boundingBox
        let texts = candidates.map { tsvSafe($0.string) }
        let padded = texts + Array(repeating: "", count: max(0, 3 - texts.count))
        out.append(String(format: "%.5f\t%.5f\t%.5f\t%.5f\t%.6f\t%@\t%@\t%@",
                          b.origin.x, b.origin.y, b.size.width, b.size.height,
                          best.confidence, padded[0], padded[1], padded[2]))
    }
    print(out.joined(separator: "\n"))
}
req.recognitionLevel = .accurate
req.usesLanguageCorrection = true
req.recognitionLanguages = ["th-TH", "en-US"]

let handler = VNImageRequestHandler(cgImage: cg, options: [:])
do { try handler.perform([req]) } catch { print("ERR perform: \(error)") }
