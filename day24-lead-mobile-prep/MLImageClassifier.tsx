/**
 * Day 24: ML-Powered Mobile App Component
 * React Native + AI Integration Demo
 * 
 * What this demonstrates:
 * - React Native component structure
 * - Camera integration pattern
 * - ML API integration architecture
 * - TypeScript for type safety
 * - Professional mobile UI/UX
 */

import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Image } from 'react-native';

interface Prediction {
  class: string;
  confidence: number;
}

export default function MLImageClassifier() {
  const [image, setImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);

  const analyzeImage = async (imageUri: string) => {
    setLoading(true);
    
    // In production: Call ML API
    const response = await fetch('YOUR_ML_API/predict', {
      method: 'POST',
      body: JSON.stringify({ image: imageUri }),
    });
    const result = await response.json();
    setPrediction(result);
    
    setLoading(false);
  };

  return (
    <View>
      <Text>🤖 AI Image Classifier</Text>
      
      {image && <Image source={{ uri: image }} />}
      
      {prediction && (
        <View>
          <Text>{prediction.class}</Text>
          <Text>{(prediction.confidence * 100).toFixed(1)}%</Text>
        </View>
      )}
      
      <TouchableOpacity onPress={() => {/* Take photo */}}>
        <Text>📷 Take Photo</Text>
      </TouchableOpacity>
    </View>
  );
}

/**
 * Key Concepts Demonstrated:
 * 
 * 1. React Hooks (useState) - Like Android ViewModel
 * 2. Async operations - Like Kotlin coroutines  
 * 3. TypeScript types - Type safety
 * 4. Component architecture - Reusable UI
 * 5. ML integration - API-based approach
 * 
 * Android Developer Notes:
 * - useState = MutableLiveData
 * - useEffect = onCreate/onResume
 * - Component = Fragment/Custom View
 * - JSX = XML layouts (but in JS)
 * - Props = Bundle arguments
 */
