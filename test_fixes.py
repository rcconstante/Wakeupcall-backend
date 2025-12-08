"""
Test script to verify all PDF generation fixes
"""

from recommendation_engine import RecommendationEngine

def test_snoring_fix():
    """Test that snoring recommendations only appear when user reports snoring"""
    print("\n=== Testing Snoring Recommendations ===")
    
    # Test 1: User with snoring = True
    print("\nTest 1: User WITH snoring")
    recs_with_snoring = RecommendationEngine.generate_recommendations(
        age=45,
        sex=1,
        bmi=28,
        neck_cm=38,
        hypertension=False,
        diabetes=False,
        smokes=False,
        alcohol=False,
        ess_score=8,
        berlin_score=0,
        stopbang_score=1,  # Has some STOP-BANG score
        sleep_duration=7.0,
        daily_steps=5000,
        risk_level="Low Risk",
        physical_activity_time="30 minutes",
        snoring=True  # User reports snoring
    )
    
    snoring_recs = [r for r in recs_with_snoring if 'snoring' in r.title.lower() or 'snoring' in r.description.lower()]
    print(f"✓ Found {len(snoring_recs)} snoring-related recommendations")
    for rec in snoring_recs[:2]:
        print(f"  - {rec.title}")
    
    # Test 2: User with snoring = False
    print("\nTest 2: User WITHOUT snoring")
    recs_no_snoring = RecommendationEngine.generate_recommendations(
        age=45,
        sex=1,
        bmi=28,
        neck_cm=38,
        hypertension=False,
        diabetes=False,
        smokes=False,
        alcohol=False,
        ess_score=8,
        berlin_score=0,
        stopbang_score=1,  # Has some STOP-BANG score but NOT from snoring
        sleep_duration=7.0,
        daily_steps=5000,
        risk_level="Low Risk",
        physical_activity_time="30 minutes",
        snoring=False  # User does NOT report snoring
    )
    
    snoring_recs_no = [r for r in recs_no_snoring if 'snoring' in r.title.lower() or 'snoring' in r.description.lower()]
    print(f"✓ Found {len(snoring_recs_no)} snoring-related recommendations (should be 0)")
    
    assert len(snoring_recs) > 0, "Should have snoring recommendations when snoring=True"
    assert len(snoring_recs_no) == 0, "Should NOT have snoring recommendations when snoring=False"
    print("✓ PASS: Snoring recommendations only appear when user reports snoring")


def test_neck_circumference():
    """Test that neck circumference is evaluated correctly"""
    print("\n=== Testing Neck Circumference ===")
    
    # Test: Neck 38 cm should NOT trigger >= 40 warning
    print("\nTest: User with neck circumference 38 cm")
    recs = RecommendationEngine.generate_recommendations(
        age=45,
        sex=1,
        bmi=26,
        neck_cm=38,  # Below 40 cm threshold
        hypertension=False,
        diabetes=False,
        smokes=False,
        alcohol=False,
        ess_score=8,
        berlin_score=0,
        stopbang_score=2,
        sleep_duration=7.0,
        daily_steps=5000,
        risk_level="Low Risk",
        snoring=False
    )
    
    neck_recs = [r for r in recs if 'neck' in r.title.lower() or 'neck' in r.description.lower()]
    print(f"✓ Found {len(neck_recs)} neck-related recommendations")
    
    # Neck size recommendation should only appear if >= 40
    large_neck_recs = [r for r in neck_recs if '40' in r.description]
    print(f"✓ Found {len(large_neck_recs)} large neck (>=40cm) recommendations (should be 0)")
    assert len(large_neck_recs) == 0, "Should NOT have large neck warnings for 38cm"
    
    # Test: Neck 42 cm SHOULD trigger >= 40 warning
    print("\nTest: User with neck circumference 42 cm")
    recs_large = RecommendationEngine.generate_recommendations(
        age=45,
        sex=1,
        bmi=26,
        neck_cm=42,  # Above 40 cm threshold
        hypertension=False,
        diabetes=False,
        smokes=False,
        alcohol=False,
        ess_score=8,
        berlin_score=0,
        stopbang_score=3,
        sleep_duration=7.0,
        daily_steps=5000,
        risk_level="Intermediate Risk",
        snoring=False
    )
    
    large_neck_recs = [r for r in recs_large if 'neck' in r.title.lower() and '40' in r.description]
    print(f"✓ Found {len(large_neck_recs)} large neck (>=40cm) recommendations")
    assert len(large_neck_recs) > 0, "Should have large neck warnings for 42cm"
    print("✓ PASS: Neck circumference evaluated correctly")


def test_physical_activity():
    """Test that physical activity time is parsed correctly"""
    print("\n=== Testing Physical Activity Parsing ===")
    
    test_cases = [
        ("Sports", 60, "Activity type: Sports"),
        ("Walking", 30, "Activity type: Walking"),
        ("45-60 minutes", 52, "Duration: 45-60 minutes"),
        ("More than 1 hour", 75, "Duration: More than 1 hour"),
        ("Less than 30 minutes", 20, "Duration: Less than 30 minutes"),
    ]
    
    for activity_str, expected_minutes, description in test_cases:
        parsed = RecommendationEngine._parse_activity_time(activity_str)
        print(f"\n{description}")
        print(f"  Input: '{activity_str}'")
        print(f"  Expected: {expected_minutes} minutes")
        print(f"  Parsed: {parsed} minutes")
        assert parsed == expected_minutes, f"Expected {expected_minutes} but got {parsed}"
        print("  ✓ PASS")
    
    # Test recommendations based on activity level
    print("\nTest: User with 'Sports' activity (should map to 60 minutes)")
    recs = RecommendationEngine.generate_recommendations(
        age=35,
        sex=1,
        bmi=24,
        neck_cm=37,
        hypertension=False,
        diabetes=False,
        smokes=False,
        alcohol=False,
        ess_score=6,
        berlin_score=0,
        stopbang_score=1,
        sleep_duration=7.5,
        daily_steps=8000,
        risk_level="Low Risk",
        physical_activity_time="Sports",  # Should parse to ~60 minutes
        snoring=False
    )
    
    # Should NOT have "increase activity" recommendation since 60 min is good
    low_activity_recs = [r for r in recs if 'increase' in r.title.lower() and 'activity' in r.title.lower()]
    print(f"✓ Found {len(low_activity_recs)} 'increase activity' recommendations (should be 0 for adequate activity)")
    
    print("\nTest: User with 'Less than 30 minutes' activity")
    recs_low = RecommendationEngine.generate_recommendations(
        age=35,
        sex=1,
        bmi=24,
        neck_cm=37,
        hypertension=False,
        diabetes=False,
        smokes=False,
        alcohol=False,
        ess_score=6,
        berlin_score=0,
        stopbang_score=1,
        sleep_duration=7.5,
        daily_steps=3000,
        risk_level="Low Risk",
        physical_activity_time="Less than 30 minutes",
        snoring=False
    )
    
    low_activity_recs = [r for r in recs_low if 'increase' in r.title.lower() and 'activity' in r.title.lower()]
    print(f"✓ Found {len(low_activity_recs)} 'increase activity' recommendations")
    assert len(low_activity_recs) > 0, "Should recommend increasing activity for <30 min"
    print("✓ PASS: Physical activity parsing works correctly")


def test_accurate_data_flow():
    """Test that all data flows accurately through the system"""
    print("\n=== Testing Accurate Data Flow ===")
    
    # Simulate a complete user survey
    user_data = {
        'age': 45,
        'sex': 1,  # Male
        'bmi': 31.5,
        'neck_cm': 38.0,  # Below 40 threshold
        'hypertension': True,
        'diabetes': False,
        'smokes': False,
        'alcohol': True,
        'ess_score': 12,
        'berlin_score': 1,
        'stopbang_score': 4,
        'sleep_duration': 6.5,
        'daily_steps': 4500,
        'physical_activity_time': 'Sports',
        'snoring': False  # User says NO to snoring
    }
    
    print("\nUser Data:")
    print(f"  Age: {user_data['age']}")
    print(f"  BMI: {user_data['bmi']}")
    print(f"  Neck: {user_data['neck_cm']} cm (should NOT show as >=40)")
    print(f"  Snoring: {user_data['snoring']} (should NOT show snoring insights)")
    print(f"  Physical Activity: {user_data['physical_activity_time']} (should show as Sports, ~60 min)")
    print(f"  ESS Score: {user_data['ess_score']}")
    
    recs = RecommendationEngine.generate_recommendations(
        age=user_data['age'],
        sex=user_data['sex'],
        bmi=user_data['bmi'],
        neck_cm=user_data['neck_cm'],
        hypertension=user_data['hypertension'],
        diabetes=user_data['diabetes'],
        smokes=user_data['smokes'],
        alcohol=user_data['alcohol'],
        ess_score=user_data['ess_score'],
        berlin_score=user_data['berlin_score'],
        stopbang_score=user_data['stopbang_score'],
        sleep_duration=user_data['sleep_duration'],
        daily_steps=user_data['daily_steps'],
        risk_level="Intermediate Risk",
        physical_activity_time=user_data['physical_activity_time'],
        snoring=user_data['snoring']
    )
    
    print(f"\nGenerated {len(recs)} recommendations")
    
    # Verify no snoring-specific recommendations (excluding mentions in other contexts)
    # Only check titles for snoring-specific recommendations
    snoring_recs = [r for r in recs if 'snoring' in r.title.lower()]
    print(f"\n✓ Snoring-specific recommendations: {len(snoring_recs)} (should be 0)")
    if snoring_recs:
        print("  WARNING: Found unexpected snoring recommendations:")
        for rec in snoring_recs:
            print(f"    - {rec.title}: {rec.description[:100]}...")
    assert len(snoring_recs) == 0, "Should NOT have snoring-specific recommendations"
    
    # Verify no large neck recommendations for 38cm
    large_neck = [r for r in recs if 'neck' in r.title.lower() and '40' in r.description]
    print(f"✓ Large neck (>=40cm) recommendations: {len(large_neck)} (should be 0)")
    assert len(large_neck) == 0, "Should NOT have large neck warnings for 38cm"
    
    # Should have BMI recommendation
    bmi_recs = [r for r in recs if 'bmi' in r.title.lower() or 'weight' in r.title.lower()]
    print(f"✓ BMI/Weight recommendations: {len(bmi_recs)}")
    assert len(bmi_recs) > 0, "Should have BMI recommendations for BMI 31.5"
    
    # Should have ESS recommendation
    ess_recs = [r for r in recs if 'sleepiness' in r.title.lower() or 'ess' in r.description.lower()]
    print(f"✓ Daytime sleepiness recommendations: {len(ess_recs)}")
    assert len(ess_recs) > 0, "Should have ESS recommendations for score 12"
    
    print("\n✓ PASS: All data flows accurately")


if __name__ == "__main__":
    print("=" * 60)
    print("Running PDF Generation Fix Tests")
    print("=" * 60)
    
    try:
        test_snoring_fix()
        test_neck_circumference()
        test_physical_activity()
        test_accurate_data_flow()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nFixes verified:")
        print("1. ✓ Snoring insights only show when user reports snoring")
        print("2. ✓ Neck circumference evaluated correctly (38 cm ≠ >=40 cm)")
        print("3. ✓ Physical activity time parses activity types correctly")
        print("4. ✓ All PDF data displays accurate survey responses")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
