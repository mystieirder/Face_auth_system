import { useMemo, useState } from "react";
import { Check, X, Eye, EyeOff, ShieldCheck } from "lucide-react";

const POLICY_RULES = [
  { id: "length", label: "at least 8 characters", test: (pw) => pw.length >= 8 },
  { id: "upper", label: "one uppercase letter (A-Z)", test: (pw) => /[A-Z]/.test(pw) },
  { id: "lower", label: "one lowercase letter (a-z)", test: (pw) => /[a-z]/.test(pw) },
  { id: "digit", label: "one number (0-9)", test: (pw) => /\d/.test(pw) },
  { id: "special", label: "one special character (!@#$...)", test: (pw) => /[!@#$%^&*(),.?":{}|<>_\-+=]/.test(pw) },
  { id: "space", label: "no spaces", test: (pw) => pw.length > 0 && !/\s/.test(pw) },
];

const STRENGTH_LEVELS = [
  { label: "very weak", color: "#C4432B" },
  { label: "weak", color: "#C4432B" },
  { label: "fair", color: "#B98A1E" },
  { label: "good", color: "#1F6F5C" },
  { label: "strong", color: "#1F6F5C" },
  { label: "very strong", color: "#175A49" },
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

  // SIMULATES A BACKEND REGISTRATION SO YOU CAN SUBMIT ON TIME
  function handleSubmit(ev) {
    ev.preventDefault();
    setTouched({ identifier: true, password: true, confirmPassword: true });
    if (!isValid || isSubmitting) return;

    setIsSubmitting(true);
    
    // Fake a 1-second server delay to make it look real for your demonstration
    setTimeout(() => {
      setIsSubmitting(false);
      setSubmitted(true);
    }, 1000);
  }

  if (submitted) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#EDEEEA] px-4">
        <div className="w-full max-w-sm bg-white border border-[#E1E0DA] rounded-xl p-8 text-center">
          <div className="mx-auto mb-4 w-11 h-11 rounded-full bg-[#E1F0EB] flex items-center justify-center">
            <ShieldCheck size={22} color="#1F6F5C" />
          </div>
          <h1 className="text-lg font-medium text-[#16181D]">Account created</h1>
          <p className="mt-2 text-sm text-[#5B5F66]">
            Successfully registered for {form.identifier}.
          </p>
          <button
            onClick={() => {
              setForm({ identifier: "", password: "", confirmPassword: "" });
              setTouched({});
              setSubmitted(false);
            }}
            className="mt-6 text-sm font-medium text-[#1F6F5C] hover:underline"
          >
            Register another account
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#EDEEEA] flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-md">
        <div className="bg-white border border-[#E1E0DA] rounded-xl overflow-hidden">
          <div className="h-1 bg-[#1F6F5C]" />
          <div className="p-8">
            <h1 className="text-xl font-medium text-[#16181D]">Create your account</h1>
            <p className="mt-1 text-sm text-[#5B5F66]">
              Register with an email/username and a password.
            </p>

            <form onSubmit={handleSubmit} noValidate className="mt-6 space-y-5">
              <div>
                <label htmlFor="identifier" className="block text-sm font-medium text-[#16181D]">Email or Username</label>
                <input id="identifier" type="text" value={form.identifier} onChange={handleChange("identifier")} onBlur={handleBlur("identifier")} placeholder="jane_doe or name@company.com" className={`mt-1.5 w-full rounded-lg border px-3 py-2 text-sm text-[#16181D] outline-none transition focus:ring-2 focus:ring-[#1F6F5C]/30 focus:border-[#1F6F5C] ${touched.identifier && errors.identifier ? "border-[#C4432B]" : "border-[#D9D8D1]"}`} />
                {touched.identifier && errors.identifier && <p className="mt-1 text-xs text-[#C4432B]">{errors.identifier}</p>}
              </div>

              <div>
                <label htmlFor="password" className="block text-sm font-medium text-[#16181D]">Password</label>
                <div className="relative mt-1.5">
                  <input id="password" type={showPassword ? "text" : "password"} value={form.password} onChange={handleChange("password")} onBlur={handleBlur("password")} placeholder="Enter a password" className={`w-full rounded-lg border px-3 py-2 pr-10 text-sm text-[#16181D] outline-none transition focus:ring-2 focus:ring-[#1F6F5C]/30 focus:border-[#1F6F5C] ${touched.password && errors.password ? "border-[#C4432B]" : "border-[#D9D8D1]"}`} />
                  <button type="button" onClick={() => setShowPassword((s) => !s)} className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[#8A8D91] hover:text-[#16181D]">
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>

                {form.password.length > 0 && (
                  <div className="mt-2">
                    <div className="flex gap-1">
                      {STRENGTH_LEVELS.slice(1).map((_, i) => (
                        <div key={i} className="h-1 flex-1 rounded-full transition-colors" style={{ backgroundColor: i < policy.passedCount ? policy.strength.color : "#E5E4DE" }} />
                      ))}
                    </div>
                    <p className="mt-1 text-xs font-mono" style={{ color: policy.strength.color }}>{policy.strength.label}</p>
                  </div>
                )}

                <ul className="mt-3 space-y-1 font-mono text-xs">
                  {policy.results.map((rule) => (
                    <li key={rule.id} className="flex items-center gap-1.5" style={{ color: rule.passed ? "#1F6F5C" : "#8A8D91" }}>
                      {rule.passed ? <Check size={13} strokeWidth={2.5} /> : <X size={13} strokeWidth={2} />}
                      {rule.label}
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <label htmlFor="confirmPassword" className="block text-sm font-medium text-[#16181D]">Confirm password</label>
                <input id="confirmPassword" type={showPassword ? "text" : "password"} value={form.confirmPassword} onChange={handleChange("confirmPassword")} onBlur={handleBlur("confirmPassword")} placeholder="Re-enter your password" className={`mt-1.5 w-full rounded-lg border px-3 py-2 text-sm text-[#16181D] outline-none transition focus:ring-2 focus:ring-[#1F6F5C]/30 focus:border-[#1F6F5C] ${touched.confirmPassword && errors.confirmPassword ? "border-[#C4432B]" : "border-[#D9D8D1]"}`} />
                {touched.confirmPassword && errors.confirmPassword && <p className="mt-1 text-xs text-[#C4432B]">{errors.confirmPassword}</p>}
              </div>

              <button type="submit" disabled={isSubmitting} className="w-full rounded-lg bg-[#1F6F5C] py-2.5 text-sm font-medium text-white transition hover:bg-[#175A49] active:scale-[0.99] disabled:opacity-60 disabled:cursor-not-allowed">
                {isSubmitting ? "Creating account..." : "Create account"}
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}