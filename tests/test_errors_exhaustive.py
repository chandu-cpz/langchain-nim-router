"""Exhaustive tests for error classification covering ALL HTTP status codes and edge cases."""
from __future__ import annotations

import pytest

from nim_router.errors import ErrorKind, classify_error


# =============================================================================
# STATUS_CODE ATTRIBUTE DETECTION
# =============================================================================


class TestStatusCodeAttribute:
    """Test classify_error when the exception has a .status_code attribute."""

    def test_400_bad_request(self):
        class E(Exception):
            status_code = 400
        assert classify_error(E("bad request")) == ErrorKind.HTTP_ERROR

    def test_401_unauthorized(self):
        class E(Exception):
            status_code = 401
        assert classify_error(E("unauthorized")) == ErrorKind.HTTP_ERROR

    def test_403_forbidden(self):
        class E(Exception):
            status_code = 403
        assert classify_error(E("forbidden")) == ErrorKind.HTTP_ERROR

    def test_404_not_found(self):
        class E(Exception):
            status_code = 404
        assert classify_error(E("not found")) == ErrorKind.MODEL_NOT_FOUND

    def test_405_method_not_allowed(self):
        class E(Exception):
            status_code = 405
        assert classify_error(E("method not allowed")) == ErrorKind.HTTP_ERROR

    def test_408_request_timeout_message_overrides_status(self):
        """FIXED: status_code=408 takes priority over 'timeout' in message."""
        class E(Exception):
            status_code = 408
        result = classify_error(E("request timeout"))
        assert result == ErrorKind.HTTP_ERROR

    def test_408_with_unrelated_message(self):
        """408 with a non-timeout message correctly returns HTTP_ERROR."""
        class E(Exception):
            status_code = 408
        assert classify_error(E("request failed")) == ErrorKind.HTTP_ERROR

    def test_409_conflict(self):
        class E(Exception):
            status_code = 409
        assert classify_error(E("conflict")) == ErrorKind.HTTP_ERROR

    def test_410_gone(self):
        class E(Exception):
            status_code = 410
        assert classify_error(E("gone")) == ErrorKind.HTTP_ERROR

    def test_413_payload_too_large(self):
        class E(Exception):
            status_code = 413
        assert classify_error(E("payload too large")) == ErrorKind.HTTP_ERROR

    def test_415_unsupported_media_type(self):
        class E(Exception):
            status_code = 415
        assert classify_error(E("unsupported media type")) == ErrorKind.HTTP_ERROR

    def test_422_unprocessable_entity(self):
        class E(Exception):
            status_code = 422
        assert classify_error(E("unprocessable")) == ErrorKind.HTTP_ERROR

    def test_429_rate_limit_with_status_code(self):
        class E(Exception):
            status_code = 429
        assert classify_error(E("too many requests")) == ErrorKind.RATE_LIMIT

    def test_429_with_unrelated_message(self):
        class E(Exception):
            status_code = 429
        assert classify_error(E("server error")) == ErrorKind.RATE_LIMIT

    def test_500_internal_server_error(self):
        class E(Exception):
            status_code = 500
        assert classify_error(E("internal server error")) == ErrorKind.HTTP_ERROR

    def test_501_not_implemented(self):
        class E(Exception):
            status_code = 501
        assert classify_error(E("not implemented")) == ErrorKind.HTTP_ERROR

    def test_502_bad_gateway(self):
        class E(Exception):
            status_code = 502
        assert classify_error(E("bad gateway")) == ErrorKind.HTTP_ERROR

    def test_503_service_unavailable(self):
        class E(Exception):
            status_code = 503
        assert classify_error(E("service unavailable")) == ErrorKind.HTTP_ERROR

    def test_504_gateway_timeout_message_overrides_status(self):
        """FIXED: status_code=504 takes priority over 'timeout' in message."""
        class E(Exception):
            status_code = 504
        result = classify_error(E("gateway timeout"))
        assert result == ErrorKind.HTTP_ERROR

    def test_504_with_unrelated_message(self):
        """504 with a non-timeout message correctly returns HTTP_ERROR."""
        class E(Exception):
            status_code = 504
        assert classify_error(E("bad gateway from upstream")) == ErrorKind.HTTP_ERROR

    def test_507_insufficient_storage(self):
        class E(Exception):
            status_code = 507
        assert classify_error(E("insufficient storage")) == ErrorKind.HTTP_ERROR

    def test_511_network_authentication_required(self):
        class E(Exception):
            status_code = 511
        assert classify_error(E("network auth required")) == ErrorKind.HTTP_ERROR

    def test_599_just_below_600(self):
        class E(Exception):
            status_code = 599
        assert classify_error(E("server error")) == ErrorKind.HTTP_ERROR

    def test_600_out_of_range(self):
        class E(Exception):
            status_code = 600
        assert classify_error(E("weird")) == ErrorKind.GENERIC

    def test_399_out_of_range(self):
        class E(Exception):
            status_code = 399
        assert classify_error(E("redirect?")) == ErrorKind.GENERIC

    def test_200_ok(self):
        class E(Exception):
            status_code = 200
        assert classify_error(E("ok")) == ErrorKind.GENERIC

    def test_100_continue(self):
        class E(Exception):
            status_code = 100
        assert classify_error(E("continue")) == ErrorKind.GENERIC


# =============================================================================
# 'code' ATTRIBUTE (alternative to status_code)
# =============================================================================


class TestCodeAttribute:
    """Test classify_error when exception uses .code instead of .status_code."""

    def test_code_429(self):
        class E(Exception):
            code = 429
        assert classify_error(E("rate limited")) == ErrorKind.RATE_LIMIT

    def test_code_404(self):
        class E(Exception):
            code = 404
        assert classify_error(E("not found")) == ErrorKind.MODEL_NOT_FOUND

    def test_code_500(self):
        class E(Exception):
            code = 500
        assert classify_error(E("server error")) == ErrorKind.HTTP_ERROR

    def test_code_400(self):
        class E(Exception):
            code = 400
        assert classify_error(E("bad request")) == ErrorKind.HTTP_ERROR

    def test_code_401(self):
        class E(Exception):
            code = 401
        assert classify_error(E("unauthorized")) == ErrorKind.HTTP_ERROR


# =============================================================================
# MESSAGE-BASED DETECTION (fallback when no status_code attribute)
# =============================================================================


class TestMessageBasedDetection:
    """Test classify_error falls back to string matching."""

    def test_rate_limit_in_message(self):
        assert classify_error(Exception("Rate limit exceeded")) == ErrorKind.RATE_LIMIT

    def test_rate_limit_lowercase(self):
        assert classify_error(Exception("rate limit")) == ErrorKind.RATE_LIMIT

    def test_too_many_requests_in_message(self):
        assert classify_error(Exception("too many requests")) == ErrorKind.RATE_LIMIT

    def test_429_in_message(self):
        assert classify_error(Exception("HTTP 429 error")) == ErrorKind.RATE_LIMIT

    def test_model_not_found_in_message(self):
        assert classify_error(Exception("model not found")) == ErrorKind.MODEL_NOT_FOUND

    def test_endpoint_not_found_in_message(self):
        assert classify_error(Exception("endpoint not found")) == ErrorKind.MODEL_NOT_FOUND

    def test_404_in_message(self):
        assert classify_error(Exception("HTTP 404 error")) == ErrorKind.MODEL_NOT_FOUND

    def test_timeout_in_message(self):
        assert classify_error(Exception("connection timeout")) == ErrorKind.TIMEOUT

    def test_timed_out_in_message(self):
        assert classify_error(Exception("request timed out")) == ErrorKind.TIMEOUT

    def test_timeout_uppercase(self):
        assert classify_error(Exception("TIMEOUT")) == ErrorKind.TIMEOUT

    def test_generic_error(self):
        assert classify_error(Exception("something went wrong")) == ErrorKind.GENERIC

    def test_empty_message(self):
        assert classify_error(Exception("")) == ErrorKind.GENERIC

    def test_no_message(self):
        assert classify_error(Exception()) == ErrorKind.GENERIC


# =============================================================================
# TIMEOUT DETECTION
# =============================================================================


class TestTimeoutDetection:
    """Test TimeoutError instance detection."""

    def test_timeout_error_instance(self):
        assert classify_error(TimeoutError()) == ErrorKind.TIMEOUT

    def test_timeout_error_with_message(self):
        assert classify_error(TimeoutError("connection timed out")) == ErrorKind.TIMEOUT

    def test_timeout_error_subclass(self):
        class MyTimeout(TimeoutError):
            pass
        assert classify_error(MyTimeout()) == ErrorKind.TIMEOUT

    def test_timeout_error_message_not_about_timeout(self):
        # TimeoutError always gets TIMEOUT regardless of message
        assert classify_error(TimeoutError("something else")) == ErrorKind.TIMEOUT


# =============================================================================
# PRIORITY: message-based checks fire before status_code range check
# =============================================================================


class TestPriorityOrdering:
    """Test actual priority: status_code==429/404 > message checks > status 400-599 > generic.

    NOTE: The implementation checks status_code==429 and status_code==404 FIRST,
    then checks message-based patterns (rate limit, timeout, model not found),
    then checks the 400-599 range. This means message content can override the
    status code for 408/504 (timeout messages) and for status 500 with '429' in msg.
    """

    def test_status_429_with_unrelated_message(self):
        class E(Exception):
            status_code = 429
        assert classify_error(E("server error")) == ErrorKind.RATE_LIMIT

    def test_status_404_with_unrelated_message(self):
        class E(Exception):
            status_code = 404
        assert classify_error(E("some other error")) == ErrorKind.MODEL_NOT_FOUND

    def test_status_500_message_says_429(self):
        """FIXED: status_code=500 takes priority over '429' in message."""
        class E(Exception):
            status_code = 500
        result = classify_error(E("429 too many requests"))
        assert result == ErrorKind.HTTP_ERROR

    def test_message_429_no_status_code(self):
        """Without status_code, message-based detection should find 429."""
        assert classify_error(Exception("error 429")) == ErrorKind.RATE_LIMIT

    def test_message_404_no_status_code(self):
        assert classify_error(Exception("error 404")) == ErrorKind.MODEL_NOT_FOUND

    def test_status_429_message_404(self):
        """status_code=429 checked first, message 404 never reached."""
        class E(Exception):
            status_code = 429
        assert classify_error(E("404 not found")) == ErrorKind.RATE_LIMIT


# =============================================================================
# BOUNDARY / EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Edge cases in error classification."""

    def test_status_code_zero(self):
        class E(Exception):
            status_code = 0
        assert classify_error(E("zero")) == ErrorKind.GENERIC

    def test_status_code_negative(self):
        class E(Exception):
            status_code = -1
        assert classify_error(E("negative")) == ErrorKind.GENERIC

    def test_status_code_none(self):
        class E(Exception):
            status_code = None
        assert classify_error(E("none")) == ErrorKind.GENERIC

    def test_code_none(self):
        class E(Exception):
            code = None
        assert classify_error(E("none")) == ErrorKind.GENERIC

    def test_both_status_code_and_code_attributes(self):
        """When both exist, status_code is checked first."""
        class E(Exception):
            status_code = 500
            code = 429
        # status_code=500 is checked first -> HTTP_ERROR
        assert classify_error(E("error")) == ErrorKind.HTTP_ERROR

    def test_non_string_status_code_is_handled(self):
        """FIXED: string status_code is converted to int, not crash."""
        class E(Exception):
            status_code = "429"
        result = classify_error(E("error"))
        assert result == ErrorKind.RATE_LIMIT

    def test_float_status_code_matches_rate_limit(self):
        """BUG: float 429.0 is truthy and passes 'if status == 429' check (Python numeric comparison)."""
        class E(Exception):
            status_code = 429.0
        # ACTUAL: 429.0 == 429 is True in Python, so this IS caught as rate_limit
        result = classify_error(E("error"))
        assert result == ErrorKind.RATE_LIMIT

    def test_bool_status_code(self):
        class E(Exception):
            status_code = True
        assert classify_error(E("error")) == ErrorKind.GENERIC

    def test_negative_status_code_not_http_error(self):
        class E(Exception):
            status_code = -404
        assert classify_error(E("error")) == ErrorKind.GENERIC

    def test_status_code_0_not_http_error(self):
        class E(Exception):
            status_code = 0
        assert classify_error(E("error")) == ErrorKind.GENERIC


# =============================================================================
# RATE LIMIT MESSAGE VARIATIONS
# =============================================================================


class TestRateLimitMessageVariations:
    """Ensure various phrasings of rate limit are caught."""

    @pytest.mark.parametrize("msg", [
        "Rate Limit",
        "rate limit exceeded",
        "You have exceeded the rate limit",
        "Too Many Requests",
        "too many requests",
        "429: too many requests",
        "HTTP 429",
        "Error 429",
        "status code 429",
    ])
    def test_rate_limit_variations(self, msg):
        assert classify_error(Exception(msg)) == ErrorKind.RATE_LIMIT

    @pytest.mark.parametrize("msg", [
        "RATE_LIMIT",
        "rate-limit",
    ])
    def test_rate_limit_variations_now_caught(self, msg):
        """FIXED: underscore/hyphen variants are now caught."""
        result = classify_error(Exception(msg))
        assert result == ErrorKind.RATE_LIMIT


# =============================================================================
# MODEL NOT FOUND MESSAGE VARIATIONS
# =============================================================================


class TestModelNotFoundVariations:
    """Ensure various phrasings of model not found are caught."""

    @pytest.mark.parametrize("msg", [
        "Model not found",
        "model not found",
        "MODEL NOT FOUND",
        "Endpoint not found",
        "endpoint not found",
        "404 not found",
        "Error 404",
        "HTTP 404",
        "status code 404",
    ])
    def test_model_not_found_variations(self, msg):
        assert classify_error(Exception(msg)) == ErrorKind.MODEL_NOT_FOUND


# =============================================================================
# TIMEOUT MESSAGE VARIATIONS
# =============================================================================


class TestTimeoutVariations:
    """Ensure various phrasings of timeout are caught."""

    @pytest.mark.parametrize("msg", [
        "timeout",
        "Timeout",
        "TIMEOUT",
        "connection timeout",
        "timed out",
        "Timed Out",
        "request timed out",
        "Read timed out",
        "Socket timeout",
    ])
    def test_timeout_variations(self, msg):
        assert classify_error(Exception(msg)) == ErrorKind.TIMEOUT


# =============================================================================
# BUGS (ALL FIXED)
# =============================================================================
# BUG 1 (FIXED): classify_error with status_code=408/504 + timeout message -> TIMEOUT not HTTP_ERROR
#   The message-based timeout check ran BEFORE the 400-599 status code range check.
#   Fix: Check status_code range BEFORE message-based checks.
#
# BUG 2 (FIXED): classify_error with status_code=500 + message containing '429' -> RATE_LIMIT not HTTP_ERROR
#   The message-based rate limit check ran BEFORE the 400-599 status code range check.
#   Fix: Check status_code range BEFORE message-based checks.
#
# BUG 3 (FIXED): classify_error crashed with TypeError when status_code is a string
#   Line 68: `if status and 400 <= status < 600` crashed when status is "429".
#   Fix: Convert status to int with try/except before comparison.
#
# BUG 4 (FIXED): Message variations 'RATE_LIMIT' and 'rate-limit' were not caught
#   The check `"rate limit" in msg` required a space, so underscore/hyphen variants failed.
#   Fix: Also check for `"rate_limit" in msg` and `"rate-limit" in msg`.
# =============================================================================
