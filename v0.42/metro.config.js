// metro.config.js
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// autoriser les .tflite comme assets
config.resolver.assetExts.push('tflite');

module.exports = config;