"use client";

import React, { useState, useRef, useEffect, DragEvent, ChangeEvent } from "react";
import { Upload, Scan, Eye, AlertTriangle, 
  CheckCircle, XCircle, FileImage, Loader2,
  Target, Shield, Search, Brain, Zap, Database, Cpu, 
  Clock} from "lucide-react";
import Image from "next/image";

// Define the type for each detection
interface Detection {
  label: string;
  confidence: number;
}

interface Metadata {
  location?: string;
  timestamp?: string;
  device?: string;
  dimensions?: string;
}

interface UploadResponse {
  detections: Detection[];
  metadata?: Metadata;
  threat_level?: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
}

const analysisSteps = [
  { icon: Upload, text: "Uploading target image...", duration: 3000 },
  { icon: Brain, text: "Initializing neural networks...", duration: 5000 },
  { icon: Scan, text: "Scanning for objects and patterns...", duration: 8000 },
  { icon: Database, text: "Cross-referencing threat database...", duration: 8000 },
  { icon: Cpu, text: "Processing deep learning models...", duration: 12000 },
  { icon: Zap, text: "Analyzing confidence scores...", duration: 6000 },
  { icon: Shield, text: "Generating intelligence report...", duration: 8000 }
];

const funFacts = [
  "AI models can identify over 10,000 different object types",
  "Neural networks process images similar to how human vision works",
  "Modern AI can detect objects smaller than 1% of image area",
  "Computer vision accuracy has improved 1000x since 2010",
  "AI can analyze facial expressions across 7 universal emotions",
  "Object detection models train on millions of labeled images",
  "Some AI models can identify fake images with 99%+ accuracy",
  "Computer vision is used in autonomous vehicles 60 times per second"
];

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [response, setResponse] = useState<UploadResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [currentFact, setCurrentFact] = useState(0);
  const [progress, setProgress] = useState(0);
  const [elapsedTime, setElapsedTime] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // Progress and step management
  useEffect(() => {
    if (!isAnalyzing) return;

    const startTime = Date.now();
    const totalDuration = 50000; // 50 seconds

    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const progressPercent = Math.min((elapsed / totalDuration) * 100, 95);
      setProgress(progressPercent);
      setElapsedTime(Math.floor(elapsed / 1000));

      // Update current step based on elapsed time
      let stepIndex = 0;
      let cumulativeDuration = 0;
      for (let i = 0; i < analysisSteps.length; i++) {
        cumulativeDuration += analysisSteps[i].duration;
        if (elapsed < cumulativeDuration) {
          stepIndex = i;
          break;
        }
      }
      setCurrentStep(stepIndex);
    }, 100);

    return () => clearInterval(interval);
  }, [isAnalyzing]);

  // Fun facts rotation
  useEffect(() => {
    if (!isAnalyzing) return;

    const factInterval = setInterval(() => {
      setCurrentFact((prev) => (prev + 1) % funFacts.length);
    }, 6000);

    return () => clearInterval(factInterval);
  }, [isAnalyzing]);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResponse(null);
    }
  };

  const handleDrag = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResponse(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setCurrentStep(0);
    setProgress(0);
    setElapsedTime(0);
    setCurrentFact(0);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/upload-image`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok){
        throw new Error(res.statusText);
      }

      const data: UploadResponse = await res.json();
      setResponse(data);
      setProgress(100);
    } catch (err) {
      console.error("Analysis error:", err);
      setResponse({
        detections: [],
        threat_level: "LOW"
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getThreatColor = (level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" | undefined) => {
    switch (level) {
      case "CRITICAL": return "text-red-400";
      case "HIGH": return "text-orange-400";
      case "MEDIUM": return "text-yellow-400";
      case "LOW": return "text-green-400";
      default: return "text-green-400";
    }
  };

  const getThreatIcon = (level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" | undefined) => {
    switch (level) {
      case "CRITICAL": 
      case "HIGH": return <AlertTriangle className="w-5 h-5" />;
      case "MEDIUM": return <Eye className="w-5 h-5" />;
      case "LOW": return <CheckCircle className="w-5 h-5" />;
      default: return <Shield className="w-5 h-5" />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 font-mono p-8 relative overflow-hidden">
      {/* Animated background grid */}
      <div className="fixed inset-0 bg-[linear-gradient(rgba(0,255,0,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(0,255,0,0.03)_1px,transparent_1px)] bg-[size:50px_50px] pointer-events-none animate-pulse" />
      
      <div className="max-w-6xl mx-auto relative z-10">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Scan className="text-green-400 w-8 h-8 animate-pulse" />
            <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-cyan-400">
              Image Analysis Terminal
            </h1>
          </div>
          <p className="text-green-300/70 max-w-2xl mx-auto">
            Deploy advanced AI reconnaissance tools for comprehensive image intelligence gathering
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="bg-black/40 border border-green-400/30 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-green-400 flex items-center gap-2 text-xl font-bold mb-6">
              <Upload className="w-5 h-5" />
              Target Acquisition
            </h2>

            {/* File Drop Zone */}
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 cursor-pointer ${
                dragActive 
                  ? "border-green-400 bg-green-400/10" 
                  : "border-green-400/50 hover:border-green-400/70"
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
              
              {previewUrl ? (
                <div className="space-y-4">
                  <Image
                    width={300}
                    height={300}
                    src={previewUrl} 
                    alt="Preview" 
                    className="max-h-40 mx-auto rounded border border-green-400/30"
                  />
                  <div className="text-green-400 text-sm">
                    <FileImage className="w-4 h-4 inline mr-2" />
                    {selectedFile?.name}
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <FileImage className="w-12 h-12 text-green-400/50 mx-auto" />
                  <div>
                    <p className="text-green-300 mb-2">Drop target image here or click to select</p>
                    <p className="text-green-400/70 text-sm">Supported formats: JPG, PNG, WEBP, GIF</p>
                  </div>
                </div>
              )}
            </div>

            {/* Analysis Button */}
            <button
              onClick={handleUpload}
              disabled={!selectedFile || isAnalyzing}
              className="w-full mt-4 bg-gradient-to-r from-green-600 to-cyan-600 hover:from-green-500 hover:to-cyan-500 text-black font-bold py-3 px-4 rounded disabled:opacity-50 transition-all duration-300"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin inline" />
                  Analyzing Target...
                </>
              ) : (
                <>
                  <Target className="w-4 h-4 mr-2 inline" />
                  Initiate Analysis
                </>
              )}
            </button>

            {/* Enhanced Progress Section */}
            {isAnalyzing && (
              <div className="mt-6 space-y-4">
                {/* Current Step */}
                <div className="flex items-center gap-3 p-3 bg-black/20 rounded border border-green-400/20">
                  {React.createElement(analysisSteps[currentStep].icon, {
                    className: "w-5 h-5 text-cyan-400 animate-spin"
                  })}
                  <span className="text-green-400 text-sm font-mono flex-1">
                    {analysisSteps[currentStep].text}
                  </span>
                  <span className="text-cyan-400 text-xs">
                    {elapsedTime}s
                  </span>
                </div>

                {/* Progress Bar */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-green-400">Progress</span>
                    <span className="text-cyan-400">{progress.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-black/40 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-green-400 to-cyan-400 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>

                {/* Fun Facts */}
                <div className="p-3 bg-black/10 rounded border border-cyan-400/20">
                  <div className="text-cyan-400 text-xs mb-1">💡 Did you know?</div>
                  <div className="text-green-300 text-sm font-mono leading-relaxed">
                    {funFacts[currentFact]}
                  </div>
                </div>

                {/* Time Estimate */}
                <div className="text-center text-green-400/70 text-xs">
                  <Clock className="w-3 h-3 inline mr-1" />
                  Estimated completion: {Math.max(0, 50 - elapsedTime)}s remaining
                </div>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="bg-black/40 border border-green-400/30 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-green-400 flex items-center gap-2 text-xl font-bold mb-6">
              <Search className="w-5 h-5" />
              Intelligence Report
            </h2>

            {!response ? (
              <div className="text-center py-12">
                {isAnalyzing ? (
                  <div className="space-y-4">
                    <div className="relative">
                      <Eye className="w-12 h-12 text-green-400 mx-auto animate-pulse" />
                      <div className="absolute inset-0 rounded-full border-2 border-green-400/20 animate-ping" />
                    </div>
                    <p className="text-green-400">Analysis in progress...</p>
                    <div className="text-green-400/70 text-sm">
                      Processing step {currentStep + 1} of {analysisSteps.length}
                    </div>
                  </div>
                ) : (
                  <>
                    <Eye className="w-12 h-12 text-green-400/30 mx-auto mb-4" />
                    <p className="text-green-400/70">Awaiting target for analysis</p>
                  </>
                )}
              </div>
            ) : (
              <div className="space-y-6">
                {/* Threat Level */}
                <div className="flex items-center justify-between p-3 bg-black/20 rounded border border-green-400/20">
                  <span className="text-green-300 font-mono text-sm">THREAT LEVEL</span>
                  <div className={`flex items-center gap-2 ${getThreatColor(response.threat_level)}`}>
                    {getThreatIcon(response.threat_level)}
                    <span className="font-bold">{response.threat_level || "LOW"}</span>
                  </div>
                </div>

                {/* Detections */}
                <div>
                  <h3 className="text-green-400 font-bold mb-3 flex items-center gap-2">
                    <Target className="w-4 h-4" />
                    Detected Objects ({response.detections.length})
                  </h3>
                  {response.detections.length > 0 ? (
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {response.detections.map((detection, index) => (
                        <div
                          key={index}
                          className="flex items-center justify-between p-2 bg-black/20 rounded border border-green-400/10"
                        >
                          <span className="text-green-300 text-sm font-mono">
                            {detection.label}
                          </span>
                          <div className="text-right">
                            <div className={`text-xs font-bold ${
                              detection.confidence > 0.8 ? 'text-green-400' :
                              detection.confidence > 0.6 ? 'text-yellow-400' : 'text-orange-400'
                            }`}>
                              {(detection.confidence * 100).toFixed(1)}%
                            </div>
                            <div className="w-16 bg-black/40 rounded-full h-1 mt-1">
                              <div 
                                className={`h-1 rounded-full ${
                                  detection.confidence > 0.8 ? 'bg-green-400' :
                                  detection.confidence > 0.6 ? 'bg-yellow-400' : 'bg-orange-400'
                                }`}
                                style={{ width: `${detection.confidence * 100}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-4 text-green-400/70">
                      <XCircle className="w-6 h-6 mx-auto mb-2" />
                      No objects detected in target image
                    </div>
                  )}
                </div>

                {/* Metadata */}
                {response.metadata && (
                  <div>
                    <h3 className="text-green-400 font-bold mb-3 flex items-center gap-2">
                      <Shield className="w-4 h-4" />
                      Extracted Metadata
                    </h3>
                    <div className="space-y-2">
                      {Object.entries(response.metadata).map(([key, value]) => (
                        <div key={key} className="flex justify-between text-sm">
                          <span className="text-green-400/70 capitalize">{key}:</span>
                          <span className="text-green-300 font-mono">{value!.toString()}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Footer Status */}
        <div className="mt-12 text-center">
          <div className="flex items-center justify-center gap-2 text-green-400/70 text-sm">
            <Shield className="w-4 h-4" />
            <span className="font-mono">OSINT Terminal Ready - All systems operational</span>
          </div>
        </div>
      </div>
    </div>
  );
}