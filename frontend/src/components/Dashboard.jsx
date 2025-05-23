// src/components/Dashboard.jsx
import React, { useState } from "react";
import { Upload, BookOpen, Search, Brain, BookText, Book, HelpCircle, List } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Alert, AlertDescription, AlertTitle } from "./ui/alert";


const Dashboard = () => {
  const [activeTab, setActiveTab] = useState("upload");
  const [uploadStatus, setUploadStatus] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [mcqs, setMcqs] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [searchResults, setSearchResults] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [numQuestions, setNumQuestions] = useState(5);
  const [selectedAnswers, setSelectedAnswers] = useState({});
  const [mcqError, setMcqError] = useState('');
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState('');

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsLoading(true);
    setUploadStatus("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/upload-pdf", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Error uploading file");
      }

      const data = await response.json();
      setUploadStatus(`Success: ${data.message} (${data.num_chunks} chunks created)`);
    } catch (error) {
      console.error("Upload error:", error);
      setUploadStatus(`Error: ${error.message || "Error uploading file"}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateMCQs = async (file) => {
    if (!file) return;

    setIsLoading(true);
    setMcqError('');
    setMcqs(null);
    setSelectedAnswers({});

    const formData = new FormData();
    formData.append("file", file);
    formData.append("num_questions", numQuestions.toString());

    try {
      const response = await fetch("http://localhost:8000/generate-mcqs", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to generate MCQs");
      }

      if (!data.mcqs || data.mcqs.length === 0) {
        throw new Error("No MCQs could be generated from this document");
      }

      setMcqs(data.mcqs);
    } catch (error) {
      console.error("MCQ generation error:", error);
      setMcqError(error.message || "Error generating MCQs");
    } finally {
      setIsLoading(false);
    }
  };

  const handleAnalyzeContent = async (file) => {
    if (!file) return;
  
    setAnalysisLoading(true);
    setAnalysisError('');
    setAnalysis(null);
    
    const formData = new FormData();
    formData.append("file", file);
  
    try {
      console.log("Starting analysis of file:", file.name);  // Debug log
      
      const response = await fetch("http://localhost:8000/analyze-content", {
        method: "POST",
        body: formData,
      });
  
      const data = await response.json();
      console.log("Received data:", data);  // Debug log
  
      if (!response.ok) {
        throw new Error(data.detail || "Error analyzing content");
      }
  
      // Ensure we have the expected data structure
      if (!data || !data.abstract) {
        throw new Error("Invalid analysis response format");
      }
  
      setAnalysis(data);
    } catch (error) {
      console.error("Analysis error:", error);
      setAnalysisError(error.message || "Failed to analyze document");
    } finally {
      setAnalysisLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery) return;

    setIsLoading(true);
    try {
      const response = await fetch(
        `http://localhost:8000/search?query=${encodeURIComponent(searchQuery)}`,
        {
          method: "GET",
        }
      );

      if (!response.ok) {
        throw new Error("Error performing search");
      }

      const data = await response.json();
      setSearchResults(data);
    } catch (error) {
      console.error("Search error:", error);
      setUploadStatus("Error performing search");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Personal Document Assistant</h1>
          <p className="mt-2 text-gray-600">
            Intelligent document analysis and learning tools
          </p>
        </div>

        {/* Main Navigation */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <Card
            className={`cursor-pointer transition-all ${activeTab === 'upload' ? 'border-blue-500 shadow-lg' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            <CardHeader>
              <Upload className="w-8 h-8 text-blue-500 mb-2" />
              <CardTitle>Upload PDF</CardTitle>
              <CardDescription>Start by uploading your document</CardDescription>
            </CardHeader>
          </Card>

          <Card
            className={`cursor-pointer transition-all ${activeTab === 'mcq' ? 'border-green-500 shadow-lg' : ''}`}
            onClick={() => setActiveTab('mcq')}
          >
            <CardHeader>
              <BookOpen className="w-8 h-8 text-green-500 mb-2" />
              <CardTitle>Generate MCQs</CardTitle>
              <CardDescription>Create practice questions</CardDescription>
            </CardHeader>
          </Card>

          <Card
            className={`cursor-pointer transition-all ${activeTab === 'analyze' ? 'border-purple-500 shadow-lg' : ''}`}
            onClick={() => setActiveTab('analyze')}
          >
            <CardHeader>
              <Brain className="w-8 h-8 text-purple-500 mb-2" />
              <CardTitle>Analyze Content</CardTitle>
              <CardDescription>Get insights and summaries</CardDescription>
            </CardHeader>
          </Card>

          <Card
            className={`cursor-pointer transition-all ${activeTab === 'search' ? 'border-orange-500 shadow-lg' : ''}`}
            onClick={() => setActiveTab('search')}
          >
            <CardHeader>
              <Search className="w-8 h-8 text-orange-500 mb-2" />
              <CardTitle>Smart Search</CardTitle>
              <CardDescription>Search through your documents</CardDescription>
            </CardHeader>
          </Card>
        </div>

        {/* Content Area */}
        <Card className="w-full">
          <CardHeader>
            <CardTitle>
              {activeTab === 'upload' && 'Upload Document'}
              {activeTab === 'mcq' && 'Generate Practice Questions'}
              {activeTab === 'analyze' && 'Content Analysis'}
              {activeTab === 'search' && 'Smart Search'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {activeTab === 'upload' && (
              <div className="space-y-4">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={handleFileUpload}
                    className="hidden"
                    id="file-upload"
                  />
                  <label
                    htmlFor="file-upload"
                    className="cursor-pointer inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
                  >
                    {isLoading ? 'Processing...' : 'Choose PDF File'}
                  </label>
                  <p className="mt-2 text-sm text-gray-600">or drag and drop your file here</p>
                </div>
                {uploadStatus && (
                  <Alert className={uploadStatus.includes('Error') ? 'bg-red-50' : 'bg-green-50'}>
                    <AlertTitle>{uploadStatus.includes('Error') ? 'Error' : 'Success'}</AlertTitle>
                    <AlertDescription>{uploadStatus}</AlertDescription>
                  </Alert>
                )}
              </div>
            )}

            {activeTab === 'mcq' && (
              <div className="space-y-6">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-8">
                  <div className="max-w-md mx-auto">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Number of Questions
                    </label>
                    <select
                      className="block w-full rounded-md border-gray-300 shadow-sm p-2 mb-4"
                      value={numQuestions}
                      onChange={(e) => setNumQuestions(parseInt(e.target.value))}
                    >
                      <option value={5}>5 Questions</option>
                      <option value={10}>10 Questions</option>
                      <option value={15}>15 Questions</option>
                      <option value={20}>20 Questions</option>
                    </select>

                    <input
                      type="file"
                      accept=".pdf"
                      onChange={(e) => handleGenerateMCQs(e.target.files[0])}
                      className="hidden"
                      id="mcq-file-upload"
                    />
                    <div className="text-center">
                      <label
                        htmlFor="mcq-file-upload"
                        className="cursor-pointer inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-green-600 hover:bg-green-700"
                      >
                        {isLoading ? 'Generating Questions...' : 'Choose PDF & Generate MCQs'}
                      </label>
                      <p className="mt-2 text-sm text-gray-600">Select a PDF to generate practice questions</p>
                    </div>
                  </div>
                </div>

                {mcqError && (
                  <Alert className="bg-red-50">
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{mcqError}</AlertDescription>
                  </Alert>
                )}

                {mcqs && mcqs.length > 0 && (
                  <div className="space-y-6">
                    <div className="flex justify-between items-center">
                      <h3 className="text-lg font-medium">Practice Questions</h3>
                      <button
                        onClick={() => {
                          setMcqs(null);
                          setSelectedAnswers({});
                        }}
                        className="text-sm text-red-600 hover:text-red-800"
                      >
                        Clear Questions
                      </button>
                    </div>

                    {mcqs.map((mcq, index) => (
                      <div key={index} className="bg-white rounded-lg border shadow-sm p-6">
                        <div className="flex justify-between items-start">
                          <h4 className="text-lg font-medium mb-4">Question {index + 1}</h4>
                          <span className="text-sm text-gray-500">
                            {selectedAnswers[index] ? 'Answered' : 'Not answered'}
                          </span>
                        </div>

                        <p className="text-gray-900 mb-4">{mcq.question}</p>

                        <div className="space-y-3">
                          {mcq.options.map((option, optIndex) => {
                            const optionLabel = option.split(')')[0];
                            return (
                              <div key={optIndex} className="flex items-start space-x-3">
                                <input
                                  type="radio"
                                  id={`q${index}-opt${optIndex}`}
                                  name={`question-${index}`}
                                  value={optionLabel}
                                  onChange={() => {
                                    setSelectedAnswers(prev => ({
                                      ...prev,
                                      [index]: optionLabel
                                    }));
                                  }}
                                  className="mt-1"
                                />
                                <label
                                  htmlFor={`q${index}-opt${optIndex}`}
                                  className="text-gray-700 cursor-pointer"
                                >
                                  {option}
                                </label>
                              </div>
                            );
                          })}
                        </div>

                        {selectedAnswers[index] && (
                          <div className="mt-4 p-4 rounded-md bg-gray-50">
                            <div className="flex items-center space-x-2 mb-2">
                              <span className="font-medium">Correct Answer:</span>
                              <span className={`px-2 py-1 rounded text-sm ${selectedAnswers[index] === mcq.correct_answer
                                ? 'bg-green-100 text-green-800'
                                : 'bg-red-100 text-red-800'
                                }`}>
                                {mcq.correct_answer}
                              </span>
                            </div>
                            <p className="text-sm text-gray-600">{mcq.explanation}</p>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'analyze' && (
              <div className="space-y-6">
                {/* File Upload Section */}
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={(e) => handleAnalyzeContent(e.target.files[0])}
                    className="hidden"
                    id="analysis-file-upload"
                  />
                  <label
                    htmlFor="analysis-file-upload"
                    className="cursor-pointer inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-purple-600 hover:bg-purple-700"
                  >
                    {analysisLoading ? 'Analyzing Document...' : 'Choose PDF for Analysis'}
                  </label>
                  <p className="mt-2 text-sm text-gray-600">Select a PDF to analyze its content</p>
                </div>

                {/* Analysis Results */}
                {analysis && (
                  <div className="space-y-6">
                    {/* Abstract Section */}
                    <div className="bg-white rounded-lg shadow-sm p-6 border">
                      <div className="flex items-center mb-4">
                        <BookText className="w-6 h-6 text-blue-600 mr-2" />
                        <h3 className="text-xl font-semibold">Abstract Suggestion</h3>
                      </div>
                      <p className="text-gray-700 leading-relaxed">
                        {analysis.abstract.content}
                      </p>
                    </div>

                    {/* Chapter Suggestions */}
                    {analysis.chapterSuggestions && analysis.chapterSuggestions.length > 0 && (
                      <div className="bg-white rounded-lg shadow-sm p-6 border">
                        <div className="flex items-center mb-4">
                          <Book className="w-6 h-6 text-green-600 mr-2" />
                          <h3 className="text-xl font-semibold">Chapter Structure Suggestions</h3>
                        </div>
                        <div className="space-y-4">
                          {analysis.chapterSuggestions.map((chapter, index) => (
                            <div key={index} className="border-l-4 border-green-500 pl-4">
                              <h4 className="font-medium text-lg">{chapter.title}</h4>
                              <ul className="mt-2 space-y-1">
                                {chapter.keyPoints.map((point, idx) => (
                                  <li key={idx} className="text-gray-600">• {point}</li>
                                ))}
                              </ul>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Study Questions */}
                    {analysis.studyQuestions && analysis.studyQuestions.length > 0 && (
                      <div className="bg-white rounded-lg shadow-sm p-6 border">
                        <div className="flex items-center mb-4">
                          <HelpCircle className="w-6 h-6 text-purple-600 mr-2" />
                          <h3 className="text-xl font-semibold">Study Questions</h3>
                        </div>
                        <div className="space-y-4">
                          {analysis.studyQuestions.map((item, index) => (
                            <div key={index} className="border-l-4 border-purple-500 pl-4">
                              <div className="flex items-center">
                                <span className="text-sm font-medium text-purple-600 bg-purple-50 px-2 py-1 rounded">
                                  {item.type}
                                </span>
                              </div>
                              <p className="font-medium mt-2">{item.question}</p>
                              <p className="text-gray-600 mt-1">
                                <span className="font-medium">Suggested Focus:</span> {item.suggestedAnswer}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Key Insights */}
                    {analysis.keyInsights && analysis.keyInsights.length > 0 && (
                      <div className="bg-white rounded-lg shadow-sm p-6 border">
                        <div className="flex items-center mb-4">
                          <List className="w-6 h-6 text-orange-600 mr-2" />
                          <h3 className="text-xl font-semibold">Key Insights</h3>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {analysis.keyInsights.map((insight, index) => (
                            <div
                              key={index}
                              className="bg-orange-50 p-4 rounded-lg border border-orange-100"
                            >
                              <p className="text-gray-800">{insight}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'search' && (
              <div className="space-y-4">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Enter your search query..."
                    className="flex-1 border rounded px-3 py-2"
                  />
                  <button
                    onClick={handleSearch}
                    className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                    disabled={isLoading}
                  >
                    {isLoading ? 'Searching...' : 'Search'}
                  </button>
                </div>
                {searchResults && (
                  <div className="mt-4 space-y-4">
                    {searchResults.documents[0].map((doc, index) => (
                      <div key={index} className="border rounded p-4">
                        <p>{doc}</p>
                        <p className="text-sm text-gray-500 mt-2">
                          Source: {searchResults.metadatas[0][index].source}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;