const express = require('express');
const router = express.Router();
const predictionController = require('../controllers/prediction.controller');
const reportController = require('../controllers/report.controller');

// Prediction & Training
router.post('/predict-user', predictionController.predictUser);
router.post('/train', predictionController.train);

// Reports
router.get('/dashboard', reportController.getDashboard);
router.post('/report/generate', reportController.generateReport);
router.get('/report/download', reportController.downloadPdf);

module.exports = router;
