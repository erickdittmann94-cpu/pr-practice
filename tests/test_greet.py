import unittest

from greet import greet


class GreetTest(unittest.TestCase):
    def test_greet_returns_german_greeting(self):
        self.assertEqual(greet("Welt"), "Hallo, Welt!")

    def test_greet_uses_given_name(self):
        self.assertEqual(greet("Erick"), "Hallo, Erick!")


if __name__ == "__main__":
    unittest.main()
