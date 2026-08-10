const crypto = require('crypto');
const getRazorpayInstance = require('../config/razorpay');
const connectDB = require('../config/db');
const { getPool, sql } = connectDB;
const PLANS = require('../config/plans');

async function createOrder(req, res) {
  try {
    const { planId } = req.body;
    const plan = PLANS[planId];

    if (!plan) {
      return res.status(400).json({ success: false, message: 'Invalid plan selected' });
    }

    const razorpay = getRazorpayInstance();
    const order = await razorpay.orders.create({
      amount: plan.amount * 100,
      currency: 'INR',
      receipt: `receipt_${planId}_${Date.now()}`,
    });

    const pool = getPool();
    await pool
      .request()
      .input('planName', sql.NVarChar, plan.name)
      .input('orderId', sql.NVarChar, order.id)
      .input('amount', sql.Float, plan.amount)
      .query('INSERT INTO Payments (PlanName, RazorpayOrderId, Amount) VALUES (@planName, @orderId, @amount)');

    res.json({
      success: true,
      orderId: order.id,
      amount: order.amount,
      currency: order.currency,
      key: process.env.RAZORPAY_KEY_ID,
      planName: plan.name,
    });
  } catch (err) {
    console.error('Create order error:', err);
    res.status(500).json({ success: false, message: 'Could not create payment order' });
  }
}

async function verifyPayment(req, res) {
  try {
    const { razorpay_order_id, razorpay_payment_id, razorpay_signature } = req.body;

    const body = razorpay_order_id + '|' + razorpay_payment_id;
    const expectedSignature = crypto
      .createHmac('sha256', process.env.RAZORPAY_KEY_SECRET)
      .update(body)
      .digest('hex');

    const pool = getPool();

    if (expectedSignature !== razorpay_signature) {
      await pool
        .request()
        .input('orderId', sql.NVarChar, razorpay_order_id)
        .query("UPDATE Payments SET Status = 'failed' WHERE RazorpayOrderId = @orderId");
      return res.status(400).json({ success: false, message: 'Payment verification failed' });
    }

    await pool
      .request()
      .input('orderId', sql.NVarChar, razorpay_order_id)
      .input('paymentId', sql.NVarChar, razorpay_payment_id)
      .query("UPDATE Payments SET RazorpayPaymentId = @paymentId, Status = 'paid' WHERE RazorpayOrderId = @orderId");

    res.json({ success: true, message: 'Payment verified', orderId: razorpay_order_id });
  } catch (err) {
    console.error('Verify payment error:', err);
    res.status(500).json({ success: false, message: 'Payment verification error' });
  }
}

module.exports = { createOrder, verifyPayment };