const mlService = require('../services/ml.service');

exports.predictUser = async (req, res, next) => {
  try {
    const result = await mlService.predictUser(req.body);
    res.json(result);
  } catch (error) {
    next(error);
  }
};

exports.train = async (req, res, next) => {
  try {
    const result = await mlService.trainModel(req.body);
    res.json(result);
  } catch (error) {
    next(error);
  }
};
