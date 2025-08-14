"use client";

import { useState, useRef } from "react";
import { motion } from "framer-motion";
import { Upload, Scan, Eye, AlertTriangle, CheckCircle, XCircle, FileImage, Loader2, Target, Shield, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import Image from "next/image";

// Define the type for each detection
interface Detection {
  label: string;
  confidence: number;
}

// Define the full server response type
interface UploadResponse {
  detections: Detection[];
  metadata?: {
    location?: string;
    timestamp?: string;
    device?: string;
    dimensions?: string;
  };
  threat_level?: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
}

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [response, setResponse] = useState<UploadResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResponse(null);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
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
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/upload-image`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Analysis failed");

      const data: UploadResponse = await res.json();
      setResponse(data);
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

  const getThreatColor = (level?: string) => {
    switch (level) {
      case "CRITICAL": return "text-red-400";
      case "HIGH": return "text-orange-400";
      case "MEDIUM": return "text-yellow-400";
      case "LOW": return "text-green-400";
      default: return "text-green-400";
    }
  };

  const getThreatIcon = (level?: string) => {
    switch (level) {
      case "CRITICAL": 
      case "HIGH": return <AlertTriangle className="w-5 h-5" />;
      case "MEDIUM": return <Eye className="w-5 h-5" />;
      case "LOW": return <CheckCircle className="w-5 h-5" />;
      default: return <Shield className="w-5 h-5" />;
    }
  };

  return (
    <div className="min-h-screen bg-background font-mono p-8 relative overflow-hidden">
      {/* Background grid */}
      <div className="fixed inset-0 bg-[linear-gradient(rgba(0,255,0,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(0,255,0,0.03)_1px,transparent_1px)] bg-[size:50px_50px] pointer-events-none" />
      
      <div className="max-w-6xl mx-auto relative z-10">
        {/* Header */}
        <motion.div
          className="text-center mb-12"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex items-center justify-center gap-3 mb-4">
            <Scan className="text-green-400 w-8 h-8" />
            <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-cyan-400">
              Image Analysis Terminal
            </h1>
          </div>
          <p className="text-green-300/70 max-w-2xl mx-auto">
            Deploy advanced AI reconnaissance tools for comprehensive image intelligence gathering
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <Card className="bg-black/40 border-green-400/30 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-green-400 flex items-center gap-2">
                  <Upload className="w-5 h-5" />
                  Target Acquisition
                </CardTitle>
              </CardHeader>
              <CardContent>
                {/* File Drop Zone */}
                <div
                  className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 ${
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
                <Button
                  onClick={handleUpload}
                  disabled={!selectedFile || isAnalyzing}
                  className="w-full mt-4 bg-gradient-to-r from-green-600 to-cyan-600 hover:from-green-500 hover:to-cyan-500 text-black font-bold disabled:opacity-50"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Analyzing Target...
                    </>
                  ) : (
                    <>
                      <Target className="w-4 h-4 mr-2" />
                      Initiate Analysis
                    </>
                  )}
                </Button>

                {/* Progress Bar */}
                {isAnalyzing && (
                  <div className="mt-4 space-y-2">
                    <div className="text-green-400 text-sm font-mono">
                      <span className="text-cyan-400">{'>'}</span> Running detection algorithms...
                    </div>
                    <Progress value={75} className="bg-black/40" />
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>

          {/* Results Section */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <Card className="bg-black/40 border-green-400/30 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-green-400 flex items-center gap-2">
                  <Search className="w-5 h-5" />
                  Intelligence Report
                </CardTitle>
              </CardHeader>
              <CardContent>
                {!response ? (
                  <div className="text-center py-12">
                    <Eye className="w-12 h-12 text-green-400/30 mx-auto mb-4" />
                    <p className="text-green-400/70">Awaiting target for analysis</p>
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
                            <motion.div
                              key={index}
                              className="flex items-center justify-between p-2 bg-black/20 rounded border border-green-400/10"
                              initial={{ opacity: 0, x: -10 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: index * 0.1 }}
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
                            </motion.div>
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
                              <span className="text-green-300 font-mono">{value}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Footer Status */}
        <motion.div
          className="mt-12 text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.6 }}
        >
          <div className="flex items-center justify-center gap-2 text-green-400/70 text-sm">
            <Shield className="w-4 h-4" />
            <span className="font-mono">OSINT Terminal Ready - All systems operational</span>
          </div>
        </motion.div>
      </div>
    </div>
  );
}