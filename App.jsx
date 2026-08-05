import "./App.css";
import { useMemo, useState } from "react";
import { Check, X, Eye, EyeOff, ShieldCheck } from "lucide-react";
import "./App.css"; // Links your new custom CSS

const POLICY_RULES = [
  { id: "length", label: "at least 8 characters", test: (pw) => pw.length >= 8 },
  { id: "upper", label: "one uppercase letter (A-Z)", test: (pw) => /[A-Z]/.test(pw) },
  { id: "lower", label: "one lowercase letter (a-z)", test: (pw) => /[a-z]/.test(pw) },
  { id: "digit", label: "one number (0-9)", test: (pw) => /\d/.test(pw) },
  { id: "special", label: "one special character (!@#$...)", test: (pw) => /[!@#$%^&*(),.?":{}|<>_\-+=]/.test(pw) },
  { id: "space", label: "no spaces", test: (pw) => pw.length > 0 && !/\s/.test(pw) },
];

const STRENGTH_LEVELS = [
  { label: "very weak", color: "#EF4444" },
  { label: "weak", color: "#F59E0B" },
  { label: "fair", color: "#EAB308" },
  { label: "good", color: "#22C55E" },
  { label: "strong", color: "#14B8A6" },
  { label: "very strong", color: "#0F766E" },
];

function usePasswordPolicy(password) {
  return useMemo(() => {
    const results = POLICY_RULES.map((rule) => ({
      ...rule,
      passed: rule.test(password),
    }));
    const passedCount = results.filter((r) => r.passed).length;
    const level = password.length === 0 ? 0 : Math.max(1, passedCount);
    const strength = STRENGTH_LEVELS[Math.min(level, STRENGTH_LEVELS.length - 1)];
    const allPassed = results.every((r) => r.passed);
    return { results, passedCount, strength, allPassed };
  }, [password]);
}

export default function App() {
  const [form, setForm] = useState({ identifier: "", password: "", confirmPassword: "" });
  const [showPassword, setShowPassword] = useState(false);
  const [touched, setTouched] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const policy = usePasswordPolicy(form.password);

  const errors = useMemo(() => {
    const e = {};
    if (!form.identifier || form.identifier.length < 3) {
      e.identifier = "Enter a valid email or username.";
    }
    if (!form.password) e.password = "Password is required.";
    else if (!policy.allPassed) e.password = "Password does not meet the policy below.";

    if (!form.confirmPassword) e.confirmPassword = "Confirm your password.";
    else if (form.confirmPassword !== form.password) e.confirmPassword = "Passwords do not match.";

    return e;
  }, [form, policy.allPassed]);

  const isValid = Object.keys(errors).length === 0;

  function handleChange(field) {
    return (ev) => {
      setForm((f) => ({ ...f, [field]: ev.target.value }));
    };
  }

  function handleBlur(field) {
    return () => setTouched((t) => ({ ...t, [field]: true }));
  }

  function handleSubmit(ev) {
    ev.preventDefault();
    setTouched({ identifier: true, password: true, confirmPassword: true });
    if (!isValid || isSubmitting) return;

    setIsSubmitting(true);
    
    setTimeout(() => {
      setIsSubmitting(false);
      setSubmitted(true);
    }, 1000);
  }

  if (submitted) {
    return (
      <div className="app-wrapper">
        <div className="registration-card">
          <div className="card-header-bar" />
          <div className="card-content success-container">
            <div className="success-icon">
              <ShieldCheck size={28} strokeWidth={2} />
            </div>
            <h1 className="card-title">Welcome aboard</h1>
            <p className="card-subtitle" style={{ marginBottom: 0 }}>
              Account successfully registered for <strong>{form.identifier}</strong>.
            </p>
            <button
              onClick={() => {
                setForm({ identifier: "", password: "", confirmPassword: "" });
                setTouched({});
                setSubmitted(false);
              }}
              className="success-button"
            >
              Register another account
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-wrapper">
      <div className="registration-card">
        <div className="card-header-bar" />
        
        <div className="card-content">
          <h1 className="card-title">Create an account</h1>
          <p className="card-subtitle">Register securely with an email or username.</p>

          <form onSubmit={handleSubmit} noValidate>
            
            <div className="form-group">
              <label htmlFor="identifier" className="form-label">Email or Username</label>
              <input 
                id="identifier" 
                type="text" 
                value={form.identifier} 
                onChange={handleChange("identifier")} 
                onBlur={handleBlur("identifier")} 
                placeholder="jane_doe or name@company.com" 
                className={`form-input ${touched.identifier && errors.identifier ? "error" : ""}`} 
              />
              {touched.identifier && errors.identifier && <p className="error-text">{errors.identifier}</p>}
            </div>

            <div className="form-group">
              <label htmlFor="password" className="form-label">Password</label>
              <div className="password-wrapper">
                <input 
                  id="password" 
                  type={showPassword ? "text" : "password"} 
                  value={form.password} 
                  onChange={handleChange("password")} 
                  onBlur={handleBlur("password")} 
                  placeholder="Create a strong password" 
                  className={`form-input ${touched.password && errors.password ? "error" : ""}`} 
                />
                <button 
                  type="button" 
                  onClick={() => setShowPassword((s) => !s)} 
                  className="toggle-password"
                >
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>

              {form.password.length > 0 && (
                <div className="strength-meter">
                  <div className="strength-bars">
                    {STRENGTH_LEVELS.slice(1).map((_, i) => (
                      <div 
                        key={i} 
                        className="strength-bar" 
                        style={{ 
                          backgroundColor: i < policy.passedCount ? policy.strength.color : "#E5E7EB",
                          opacity: i < policy.passedCount ? 1 : 0.5
                        }} 
                      />
                    ))}
                  </div>
                  <p className="strength-text" style={{ color: policy.strength.color }}>
                    {policy.strength.label}
                  </p>
                </div>
              )}

              <ul className="policy-list">
                {policy.results.map((rule) => (
                  <li key={rule.id} className="policy-item" style={{ color: rule.passed ? "#0F766E" : "#9CA3AF" }}>
                    <div className={`policy-icon-wrapper ${rule.passed ? "passed" : "failed"}`}>
                      {rule.passed ? <Check size={10} strokeWidth={3} /> : <X size={10} strokeWidth={2.5} />}
                    </div>
                    {rule.label}
                  </li>
                ))}
              </ul>
            </div>

            <div className="form-group">
              <label htmlFor="confirmPassword" className="form-label">Confirm Password</label>
              <input 
                id="confirmPassword" 
                type={showPassword ? "text" : "password"} 
                value={form.confirmPassword} 
                onChange={handleChange("confirmPassword")} 
                onBlur={handleBlur("confirmPassword")} 
                placeholder="Re-enter your password" 
                className={`form-input ${touched.confirmPassword && errors.confirmPassword ? "error" : ""}`} 
              />
              {touched.confirmPassword && errors.confirmPassword && <p className="error-text">{errors.confirmPassword}</p>}
            </div>

            <button 
              type="submit" 
              disabled={isSubmitting} 
              className="submit-button"
            >
              {isSubmitting ? "Creating account..." : "Create account"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
