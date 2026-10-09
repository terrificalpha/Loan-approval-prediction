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


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.model = lm.load_model()

    def test_model_expects_our_features(self):
        self.assertEqual(list(self.model.feature_names_in_), lm.FEATURES)

    def test_notebook_example_is_approved(self):
        # Notebook cell: input_data=(1,0,0,0,0,5849,146.41,360,1,2) -> eligible
        label, score = lm.predict(
            self.model,
            **sample(
                married="No",
                applicant_income=5849,
                loan_amount=146.41,
                property_area="Urban",
            ),
        )
        self.assertEqual(label, 1)
        self.assertGreater(score, 0.5)

    def test_poor_credit_history_is_rejected(self):
        label, score = lm.predict(self.model, **sample(credit_history="No / poor (0)"))
        self.assertEqual(label, 0)
        self.assertLess(score, 0.5)

    def test_metrics_file_is_readable(self):
        metrics = lm.load_metrics()
        self.assertIsNotNone(metrics)
        for key in ("test_accuracy", "recall_approved", "recall_not_approved"):
            self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main()
