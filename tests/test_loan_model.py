import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import loan_model as lm  # noqa: E402


def sample(**overrides):
    values = dict(
        gender="Male",
        married="Yes",
        dependents="0",
        education="Graduate",
        self_employed="No",
        applicant_income=6000,
        loan_amount=120,
        loan_term=360,
        credit_history="Yes / good (1)",
        property_area="Semiurban",
    )
    values.update(overrides)
    return values


class EncodingTests(unittest.TestCase):
    """These codes must match the training notebook's LabelEncoder output."""

    def test_codes_match_notebook(self):
        self.assertEqual(lm.GENDER, {"Female": 0, "Male": 1})
        self.assertEqual(lm.EDUCATION, {"Graduate": 0, "Not Graduate": 1})
        self.assertEqual(lm.DEPENDENTS["3+"], 3)
        self.assertEqual(lm.PROPERTY_AREA, {"Rural": 0, "Semiurban": 1, "Urban": 2})

    def test_feature_order(self):
        frame = lm.build_features(**sample())
        self.assertEqual(list(frame.columns), lm.FEATURES)
        self.assertEqual(frame.shape, (1, 10))

    def test_invalid_category_rejected(self):
        with self.assertRaises(ValueError):
            lm.build_features(**sample(gender="Other"))

    def test_invalid_numbers_rejected(self):
        with self.assertRaises(ValueError):
            lm.build_features(**sample(loan_amount=0))


class GuardrailTests(unittest.TestCase):
    def test_fallback_ranges_used_without_saved_ranges(self):
        self.assertEqual(lm.load_ranges(None), lm.FALLBACK_RANGES)

    def test_saved_ranges_are_used_and_rounded_outward(self):
        saved = {"feature_ranges": {"LoanAmount": [9.0, 699.5]}}
        ranges = lm.load_ranges(saved)
        self.assertEqual(ranges["LoanAmount"], (9, 700))
        self.assertEqual(ranges["Loan_Amount_Term"], lm.FALLBACK_RANGES["Loan_Amount_Term"])

    def test_absurd_loan_triggers_repayment_warning(self):
        self.assertIsNotNone(lm.repayment_warning(25000, 800000, 12))

    def test_reasonable_loan_has_no_warning(self):
        self.assertIsNone(lm.repayment_warning(6000, 120, 360))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.model = lm.load_model()

    def test_model_expects_our_features(self):
        self.assertEqual(list(self.model.feature_names_in_), lm.FEATURES)

    def test_scores_are_probabilities(self):
        label, score = lm.predict(self.model, **sample())
        self.assertIn(label, (0, 1))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_good_credit_scores_higher_than_poor_credit(self):
        _, good = lm.predict(self.model, **sample(credit_history="Yes / good (1)"))
        _, poor = lm.predict(self.model, **sample(credit_history="No / poor (0)"))
        self.assertGreater(good, poor)

    def test_metrics_file_is_readable(self):
        metrics = lm.load_metrics()
        self.assertIsNotNone(metrics)
        for key in ("test_accuracy", "recall_approved", "recall_not_approved"):
            self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main()
