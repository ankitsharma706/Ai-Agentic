const mlService = require('../services/ml.service');

exports.getDashboard = async (req, res, next) => {
  try {
    const data = await mlService.getDashboard();
    res.json(data);
  } catch (error) {
    next(error);
  }
};

exports.generateReport = async (req, res, next) => {
  try {
    const result = await mlService.generateReport();
    res.json(result);
  } catch (error) {
    next(error);
  }
};

exports.downloadPdf = async (req, res, next) => {
  try {
    const response = await mlService.getLatestPdf();
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'attachment; filename=churn_report.pdf');
    res.send(Buffer.from(response.data, 'binary'));
  } catch (error) {
    next(error);
  }
};
