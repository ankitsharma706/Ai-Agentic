const axios = require('axios');
const dotenv = require('dotenv');

dotenv.config();

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

const mlClient = axios.create({
  baseURL: ML_SERVICE_URL,
  timeout: 30000, // ML tasks can be slow
  headers: {
    'Content-Type': 'application/json'
  }
});

module.exports = mlClient;
