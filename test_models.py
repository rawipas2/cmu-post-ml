"""
Quick test script to validate model implementations
Tests each model with dummy data
"""
import numpy as np
import torch
import config

print("="*80)
print("🧪 Testing Model Implementations")
print("="*80)

# Create dummy data
print("\n📊 Creating dummy data...")
n_samples = 100
n_features = 1000
X_dummy = np.random.randn(n_samples, n_features).astype(np.float32)
y_dummy = np.random.randint(0, 2, n_samples)

print(f"   X shape: {X_dummy.shape}")
print(f"   y shape: {y_dummy.shape}")

# Test each model
results = []

# 1. Test SVM
print("\n" + "🔵"*40)
print("Testing SVM...")
try:
    from models import svm_model
    svm = svm_model.create_model()
    svm.train(X_dummy[:80], y_dummy[:80])
    pred = svm.predict(X_dummy[80:])
    prob = svm.predict_proba(X_dummy[80:])
    print(f"✅ SVM: Predictions shape {pred.shape}, Probas shape {prob.shape}")
    results.append(("SVM", True))
except Exception as e:
    print(f"❌ SVM failed: {e}")
    results.append(("SVM", False))

# 2. Test Neural Network
print("\n" + "🔵"*40)
print("Testing Neural Network...")
try:
    from models import neural_network_model
    nn = neural_network_model.create_model(input_dim=n_features, epochs=5)
    nn.train(X_dummy[:80], y_dummy[:80])
    pred = nn.predict(X_dummy[80:])
    prob = nn.predict_proba(X_dummy[80:])
    print(f"✅ Neural Network: Predictions shape {pred.shape}, Probas shape {prob.shape}")
    results.append(("Neural Network", True))
except Exception as e:
    print(f"❌ Neural Network failed: {e}")
    results.append(("Neural Network", False))

# 3. Test Deep Learning
print("\n" + "🔵"*40)
print("Testing Deep Learning...")
try:
    from models import deep_learning_model
    dl = deep_learning_model.create_model(input_dim=n_features, epochs=5)
    dl.train(X_dummy[:80], y_dummy[:80])
    pred = dl.predict(X_dummy[80:])
    prob = dl.predict_proba(X_dummy[80:])
    print(f"✅ Deep Learning: Predictions shape {pred.shape}, Probas shape {prob.shape}")
    results.append(("Deep Learning", True))
except Exception as e:
    print(f"❌ Deep Learning failed: {e}")
    results.append(("Deep Learning", False))

# 4. Test Naive Bayes
print("\n" + "🔵"*40)
print("Testing Naive Bayes...")
try:
    from models import naive_bayes_model
    nb = naive_bayes_model.create_model()
    nb.train(np.abs(X_dummy[:80]), y_dummy[:80])
    pred = nb.predict(np.abs(X_dummy[80:]))
    prob = nb.predict_proba(np.abs(X_dummy[80:]))
    print(f"✅ Naive Bayes: Predictions shape {pred.shape}, Probas shape {prob.shape}")
    results.append(("Naive Bayes", True))
except Exception as e:
    print(f"❌ Naive Bayes failed: {e}")
    results.append(("Naive Bayes", False))

# 5. Test Bayesian Network
print("\n" + "🔵"*40)
print("Testing Bayesian Network...")
try:
    from models import bayesian_network_model
    bn = bayesian_network_model.create_model(input_dim=n_features, epochs=5)
    bn.train(X_dummy[:80], y_dummy[:80])
    pred = bn.predict(X_dummy[80:])
    prob = bn.predict_proba(X_dummy[80:])
    print(f"✅ Bayesian Network: Predictions shape {pred.shape}, Probas shape {prob.shape}")
    results.append(("Bayesian Network", True))
except Exception as e:
    print(f"❌ Bayesian Network failed: {e}")
    results.append(("Bayesian Network", False))

# 6. Test Maximum Entropy
print("\n" + "🔵"*40)
print("Testing Maximum Entropy...")
try:
    from models import maximum_entropy_model
    me = maximum_entropy_model.create_model(input_dim=n_features, epochs=5)
    me.train(X_dummy[:80], y_dummy[:80])
    pred = me.predict(X_dummy[80:])
    prob = me.predict_proba(X_dummy[80:])
    print(f"✅ Maximum Entropy: Predictions shape {pred.shape}, Probas shape {prob.shape}")
    results.append(("Maximum Entropy", True))
except Exception as e:
    print(f"❌ Maximum Entropy failed: {e}")
    results.append(("Maximum Entropy", False))

# Test Ensemble (if we have at least 2 models)
successful_models = []
for name, success in results:
    if success:
        # Re-create models for ensemble
        pass

print("\n" + "🟢"*40)
print("Testing Ensemble Stacking...")
print("⚠️  Ensemble test requires trained base models")
print("   Will be tested during full training")

# Summary
print("\n" + "="*80)
print("📋 TEST SUMMARY")
print("="*80)

for name, success in results:
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status:12} - {name}")

passed = sum(1 for _, s in results if s)
total = len(results)

print("="*80)
print(f"\nTotal: {passed}/{total} models passed")

if passed == total:
    print("\n🎉 All models working correctly!")
else:
    print(f"\n⚠️  {total - passed} model(s) failed. Check error messages above.")

print("\n💡 Next step: Run 'python train.py' to train on real data")
