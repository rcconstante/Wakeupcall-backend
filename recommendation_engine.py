"""
Comprehensive recommendation engine for sleep apnea risk assessment.
Implements evidence-based recommendations based on CDC, AASM, NSF, WHO, and other sources.
"""

from typing import List, Dict, Optional


class Recommendation:
    """Data class for a single recommendation."""
    
    def __init__(self, title: str, description: str, source: str, priority: int = 0):
        self.title = title
        self.description = description
        self.source = source
        self.priority = priority
    
    def __repr__(self):
        return f"{self.title}: {self.description} [{self.source}]"


class RecommendationEngine:
    """
    Generates comprehensive, evidence-based recommendations for sleep apnea risk
    based on multiple factors including demographics, medical history, survey scores,
    and physical activity data.
    """
    
    @staticmethod
    def generate_recommendations(
        age: int,
        sex: int,  # 1=male, 0=female
        bmi: float,
        neck_cm: float,
        hypertension: bool,
        diabetes: bool,
        smokes: bool,
        alcohol: bool,
        ess_score: int,
        berlin_score: int,
        stopbang_score: int,
        sleep_duration: float,
        daily_steps: int = 5000,
        risk_level: str = "Unknown",
        physical_activity_time: str = None,
        physical_activity_minutes: int = None,
        snoring: bool = False
    ) -> List[Recommendation]:
        """Generate all applicable recommendations based on user data."""
        
        recommendations = []
        
        # Use physical_activity_minutes if provided (user's direct input), otherwise calculate from steps
        if physical_activity_minutes is not None:
            # Use the actual minutes entered by user
            activity_mins = physical_activity_minutes
            print(f"📊 Using user-provided physical activity minutes: {activity_mins}")
        else:
            # Fallback: calculate from daily steps
            activity_mins = RecommendationEngine._calculate_activity_minutes(daily_steps)
            print(f"📊 Calculated physical activity minutes from steps ({daily_steps}): {activity_mins}")
        
        activity_type = RecommendationEngine._determine_activity_type(activity_mins)
        activity_time = "morning"  # Default
        
        # ============================================================
        # 1. SINGLE-FACTOR RULES
        # ============================================================
        
        # Sleep Duration
        if sleep_duration < 7:
            recommendations.append(Recommendation(
                title="Increase Your Total Sleep Time",
                description="You reported sleeping less than 7 hours per night. Adults typically need 7â€“9 hours of sleep. Gradually move your bedtime earlier by about 15 minutes every few days to reduce sleep debt.",
                source="Centers for Disease Control and Prevention (CDC) â€“ Sleep Duration Recommendations; American Academy of Sleep Medicine (AASM).",
                priority=8
            ))
        
        if sleep_duration >= 9:
            recommendations.append(Recommendation(
                title="Monitor Oversleeping and Sleep Quality",
                description="You reported sleeping 9 hours or more. Oversleeping can sometimes reflect poor sleep quality or fragmented sleep. Pay attention to how refreshed you feel during the day.",
                source="American Academy of Sleep Medicine (AASM) â€“ Sleep Quality Guidance.",
                priority=5
            ))
        
        # Snoring
        if snoring:
            recommendations.append(Recommendation(
                title="Manage Snoring and Airway Obstruction",
                description="You reported regular snoring, which can be a sign of partial airway obstruction during sleep. Side-sleeping, using a supportive pillow, and avoiding heavy meals close to bedtime may help reduce snoring.",
                source="National Sleep Foundation â€“ Snoring and Sleep; American Academy of Sleep Medicine (AASM) â€“ Snoring and OSA.",
                priority=7
            ))
        
        # Epworth Sleepiness Score
        if ess_score >= 11:
            recommendations.append(Recommendation(
                title="Address Excessive Daytime Sleepiness",
                description="Your Epworth Sleepiness Score is elevated, which suggests excessive daytime sleepiness. This often reflects poor sleep quality or fragmented sleep at night.",
                source="Johns MW, Epworth Sleepiness Scale (1991); AASM â€“ Daytime Sleepiness Guidance.",
                priority=9
            ))
        
        # BMI
        if bmi >= 30:
            recommendations.append(Recommendation(
                title="Consider Weight's Impact on Breathing",
                description="Your BMI falls in a range that can increase narrowing of the upper airway during sleep. Even modest weight changes may help improve breathing and sleep quality over time.",
                source="World Health Organization (WHO) â€“ BMI Classification; AASM â€“ Obesity and OSA Risk.",
                priority=8
            ))
        
        # Neck Circumference
        if neck_cm >= 40:
            recommendations.append(Recommendation(
                title="Neck Size and Airway Narrowing",
                description="A neck circumference of 40 cm or more is associated with a higher chance of airway narrowing during sleep, which can contribute to snoring or sleep apnea.",
                source="Chung F. et al., STOP-Bang Questionnaire Guidelines.",
                priority=7
            ))
        
        # Hypertension
        if hypertension:
            recommendations.append(Recommendation(
                title="Hypertension and Sleep-Disordered Breathing",
                description="You reported hypertension. High blood pressure is commonly linked with undiagnosed sleep-disordered breathing and may be worsened by poor sleep.",
                source="American Heart Association (AHA) â€“ OSA and Hypertension.",
                priority=8
            ))
        
        # Diabetes
        if diabetes:
            recommendations.append(Recommendation(
                title="Diabetes and Sleep Quality",
                description="You reported diabetes. Blood sugar imbalance is often associated with disrupted sleep patterns, and sleep apnea is more frequent among people with diabetes.",
                source="American Diabetes Association (ADA); AASM â€“ Sleep and Metabolic Health.",
                priority=7
            ))
        
        # Alcohol
        if alcohol:
            recommendations.append(Recommendation(
                title="Reduce Alcohol Intake Near Bedtime",
                description="Since you reported alcohol use, especially if taken in the evening, it can relax the upper airway muscles, worsen snoring, and increase breathing pauses during sleep. Try to avoid alcohol at least 3â€“4 hours before bed.",
                source="American Academy of Sleep Medicine (AASM) â€“ Alcohol and Sleep Quality.",
                priority=6
            ))
        
        # STOP-BANG
        if stopbang_score >= 5:
            recommendations.append(Recommendation(
                title="High STOP-Bang Score and OSA Risk",
                description="Your STOP-Bang score falls in a range associated with higher risk of obstructive sleep apnea. Monitoring your nighttime symptoms and daytime sleepiness is especially important.",
                source="Chung F. et al., STOP-Bang Questionnaire Validation Studies.",
                priority=10
            ))
        
        # Physical Activity
        if activity_mins < 30:
            recommendations.append(Recommendation(
                title="Increase Daily Physical Activity",
                description="You reported less than 30 minutes of physical activity per day. Increasing daily movement to at least 30 minutes can help improve sleep quality, reduce sleep latency, and support overall health.",
                source="CDC Physical Activity Guidelines; Harvard Medical School â€“ Division of Sleep Medicine (Exercise and Sleep).",
                priority=7
            ))
        
        if activity_mins >= 150:
            recommendations.append(Recommendation(
                title="You Meet Activity Recommendations",
                description="Your reported activity matches or exceeds commonly recommended weekly activity levels. Regular movement is associated with deeper sleep and better daytime energy.",
                source="CDC Physical Activity Guidelines; Sleep Foundation â€“ Exercise and Sleep Quality.",
                priority=3
            ))
        
        # Activity Type
        if activity_type == "light":
            recommendations.append(Recommendation(
                title="Light Activity and Sleep Support",
                description="You indicated mostly light activity. While light movement supports general health, adding some moderate-intensity exercise may have a stronger positive effect on sleep depth and quality.",
                source="Sleep Foundation â€“ Exercise Intensity and Sleep.",
                priority=5
            ))
        elif activity_type == "moderate":
            recommendations.append(Recommendation(
                title="Moderate Exercise and Deeper Sleep",
                description="Your activity level is in the moderate range. Regular moderate exercise is associated with improved deep sleep and reduced daytime fatigue.",
                source="American Academy of Sleep Medicine (AASM) â€“ Physical Activity and Sleep.",
                priority=4
            ))
        elif activity_type == "vigorous":
            recommendations.append(Recommendation(
                title="Timing Vigorous Exercise Wisely",
                description="You reported vigorous activity. Vigorous exercise can benefit sleep overall, but if done too close to bedtime, it may temporarily increase alertness and make it harder to fall asleep.",
                source="National Sleep Foundation â€“ Vigorous Exercise and Sleep Onset.",
                priority=5
            ))
        
        # ============================================================
        # 2. TWO-FACTOR COMBINATION RULES
        # ============================================================
        
        # Alcohol + Hypertension
        if alcohol and hypertension:
            recommendations.append(Recommendation(
                title="Alcohol and Hypertension During Sleep",
                description="Combining alcohol use with hypertension can increase cardiovascular strain and contribute to more unstable breathing during sleep. Reducing evening alcohol intake can be especially beneficial for blood pressure and sleep.",
                source="American Academy of Sleep Medicine (AASM); American Heart Association (AHA).",
                priority=9
            ))
        
        # Snoring + Hypertension
        if snoring and hypertension:
            recommendations.append(Recommendation(
                title="Snoring and Blood Pressure Risk",
                description="Snoring together with high blood pressure may increase strain on your heart and blood vessels during sleep. This pattern is often seen in individuals with undiagnosed sleep apnea.",
                source="American Heart Association (AHA); AASM â€“ OSA and Cardiovascular Risk.",
                priority=9
            ))
        
        # Snoring + High BMI
        if snoring and bmi >= 30:
            recommendations.append(Recommendation(
                title="Snoring and Weight-Related Airway Narrowing",
                description="Snoring combined with a higher BMI increases the likelihood that your upper airway becomes narrowed or collapses during sleep, contributing to louder snoring or breathing pauses.",
                source="AASM â€“ Obstructive Sleep Apnea Risk Factors; WHO â€“ Obesity and Respiratory Function.",
                priority=9
            ))
        
        # Snoring + High ESS
        if snoring and ess_score >= 11:
            recommendations.append(Recommendation(
                title="Snoring and Excessive Daytime Sleepiness",
                description="Snoring plus significant daytime sleepiness suggests that your sleep may be fragmented or non-restorative, possibly due to repeated airway obstruction during the night.",
                source="Epworth Sleepiness Scale (Johns, 1991); AASM â€“ Snoring and Sleep Fragmentation.",
                priority=10
            ))
        
        # High BMI + Large Neck
        if bmi >= 30 and neck_cm >= 40:
            recommendations.append(Recommendation(
                title="Body Habitus and Airway Structure",
                description="A combination of higher BMI and larger neck circumference is strongly associated with upper airway narrowing, which increases the risk of obstructed breathing during sleep.",
                source="STOP-Bang Guidelines (Chung F. et al.); WHO â€“ BMI and OSA.",
                priority=9
            ))
        
        # Low Sleep + High ESS
        if sleep_duration < 7 and ess_score >= 11:
            recommendations.append(Recommendation(
                title="Sleep Debt and Daytime Sleepiness",
                description="Short sleep combined with elevated daytime sleepiness suggests that you are accumulating sleep debt and your sleep is not fully restorative.",
                source="CDC â€“ Sleep Duration and Health; Epworth Sleepiness Scale (Johns, 1991).",
                priority=10
            ))
        
        # High STOP-BANG + Large Neck
        if stopbang_score >= 5 and neck_cm >= 40:
            recommendations.append(Recommendation(
                title="High-Risk Screening and Neck Anatomy",
                description="Your screening score and neck circumference together suggest a high likelihood of airway narrowing during sleep, which is characteristic of obstructive sleep apnea.",
                source="Chung F. et al., STOP-Bang Questionnaire Clinical Pathways.",
                priority=10
            ))
        
        # Hypertension + Diabetes
        if hypertension and diabetes:
            recommendations.append(Recommendation(
                title="Metabolic and Blood Pressure Risks During Sleep",
                description="The combination of hypertension and diabetes is frequently seen in people with sleep-disordered breathing. Improving sleep quality can support overall cardiometabolic health.",
                source="American Heart Association (AHA); American Diabetes Association (ADA).",
                priority=9
            ))
        
        # Alcohol + Snoring
        if alcohol and snoring:
            recommendations.append(Recommendation(
                title="Alcohol's Effect on Snoring",
                description="Alcohol relaxes the muscles in the throat and can significantly worsen snoring intensity and frequency. Avoiding alcohol close to bedtime may reduce snoring.",
                source="American Academy of Sleep Medicine (AASM) â€“ Alcohol and Airway Tone.",
                priority=8
            ))
        
        # Low Activity + High ESS
        if activity_mins < 30 and ess_score >= 11:
            recommendations.append(Recommendation(
                title="Low Movement and Daytime Sleepiness",
                description="Low daily physical activity combined with significant daytime sleepiness suggests you may benefit from gradually increasing your activity to support better sleep and alertness.",
                source="CDC Physical Activity Guidelines; ESS Research on Fatigue.",
                priority=8
            ))
        
        # High BMI + Low Activity
        if bmi >= 30 and activity_mins < 30:
            recommendations.append(Recommendation(
                title="Weight and Inactivity Effects on Breathing",
                description="Higher body weight combined with low activity levels can contribute to reduced respiratory function and airway narrowing during sleep. Gradual increases in movement can be beneficial.",
                source="World Health Organization (WHO); AASM â€“ Weight, Activity, and OSA.",
                priority=9
            ))
        
        # Evening Exercise + Short Sleep
        if activity_time == "evening" and sleep_duration < 7:
            recommendations.append(Recommendation(
                title="Adjusting Evening Exercise to Improve Sleep",
                description="Since you are sleeping less than 7 hours and often exercise in the evening, shifting some workouts earlier in the day may help you wind down more easily at night.",
                source="Harvard Medical School â€“ Division of Sleep Medicine, Exercise Timing and Sleep.",
                priority=7
            ))
        
        # ============================================================
        # 3. THREE-FACTOR (OR MORE) HIGH-RISK RULES
        # ============================================================
        
        # Snoring + High BMI + High ESS
        if snoring and bmi >= 30 and ess_score >= 11:
            recommendations.append(Recommendation(
                title="Strong Pattern of Possible Sleep-Disordered Breathing",
                description="The combination of snoring, higher BMI, and significant daytime sleepiness strongly suggests fragmented or disrupted sleep, possibly due to repeated breathing interruptions at night.",
                source="American Academy of Sleep Medicine (AASM); ESS Research (Johns, 1991).",
                priority=11
            ))
        
        # Large Neck + High BMI + High STOP-BANG
        if neck_cm >= 40 and bmi >= 30 and stopbang_score >= 5:
            recommendations.append(Recommendation(
                title="Multiple Anatomical and Screening Indicators of OSA",
                description="Your neck size, weight, and STOP-Bang score together indicate a high probability of obstructive sleep apnea. This pattern is commonly seen in individuals with significant airway narrowing during sleep.",
                source="Chung F. et al., STOP-Bang Questionnaire Clinical Validation; WHO â€“ Obesity and OSA.",
                priority=11
            ))
        
        # Hypertension + Snoring + High ESS
        if hypertension and snoring and ess_score >= 11:
            recommendations.append(Recommendation(
                title="Cardiovascular Strain from Poor Sleep",
                description="High blood pressure combined with snoring and daytime sleepiness may indicate that your heart and blood vessels are under extra strain during sleep, often seen in people with sleep apnea.",
                source="American Heart Association (AHA); AASM â€“ OSA and Cardiovascular Outcomes.",
                priority=11
            ))
        
        # Short Sleep + High ESS + High STOP-BANG
        if sleep_duration < 7 and ess_score >= 11 and stopbang_score >= 5:
            recommendations.append(Recommendation(
                title="Sleep Debt and High Apnea Risk",
                description="Short sleep, significant daytime sleepiness, and a high STOP-Bang score together suggest that your sleep may be both insufficient and disrupted by breathing problems.",
                source="CDC â€“ Sleep Duration; Epworth Sleepiness Scale; Chung F. et al., STOP-Bang.",
                priority=11
            ))
        
        # Diabetes + Hypertension + Snoring
        if diabetes and hypertension and snoring:
            recommendations.append(Recommendation(
                title="Metabolic, Blood Pressure, and Airway Red Flags",
                description="The combination of diabetes, hypertension, and snoring is frequently observed in individuals with underlying sleep apnea. Addressing sleep quality can be an important part of overall health management.",
                source="American Heart Association (AHA); American Diabetes Association (ADA); AASM â€“ Sleep and Cardiometabolic Health.",
                priority=11
            ))
        
        # Low Activity + High BMI + Snoring
        if activity_mins < 30 and bmi >= 30 and snoring:
            recommendations.append(Recommendation(
                title="Activity, Weight, and Breathing Difficulties",
                description="Low daily movement combined with higher BMI and snoring may indicate increased airway resistance and reduced respiratory fitness. Gradual increases in physical activity can help support better breathing and sleep.",
                source="AASM â€“ OSA and Lifestyle; WHO; CDC Physical Activity Guidelines.",
                priority=10
            ))
        
        # Moderate Morning Exercise + High ESS + Short Sleep
        if activity_type == "moderate" and activity_time == "morning" and ess_score >= 11 and sleep_duration < 7:
            recommendations.append(Recommendation(
                title="Strengthening Your Sleep-Wake Cycle",
                description="You already benefit from morning moderate exercise, but your high daytime sleepiness and short sleep duration suggest your sleep-wake cycle may still be disrupted. Extending sleep time and keeping a consistent schedule can help.",
                source="Sleep Foundation â€“ Morning Exercise; Epworth Sleepiness Scale; CDC â€“ Sleep Duration.",
                priority=9
            ))
        
        # High STOP-BANG + Low Activity + Hypertension
        if stopbang_score >= 5 and activity_mins < 30 and hypertension:
            recommendations.append(Recommendation(
                title="High-Risk Profile with Low Activity",
                description="A high STOP-Bang score, low physical activity, and hypertension together indicate an increased cardiometabolic and sleep-related risk profile. Improving activity levels and sleep quality may have meaningful health benefits.",
                source="Chung F. et al., STOP-Bang; American Heart Association; CDC Physical Activity Guidelines.",
                priority=11
            ))
        
        # ============================================================
        # HIGH RISK: Add professional consultation at the top
        # ============================================================
        
        # Use ML model prediction for high risk determination
        # Only show "High Risk: Professional Sleep Evaluation" if ML model says High Risk
        is_high_risk = risk_level == "High Risk"
        
        if is_high_risk:
            recommendations.append(Recommendation(
                title="High Risk: Professional Sleep Evaluation Recommended",
                description="Based on your assessment results and machine learning analysis, you show multiple indicators strongly associated with sleep-disordered breathing. We strongly recommend consulting with a sleep specialist or healthcare provider for a comprehensive evaluation. A sleep study (polysomnography) may be necessary for accurate diagnosis and treatment planning.",
                source="American Academy of Sleep Medicine (AASM); Centers for Disease Control and Prevention (CDC); National Sleep Foundation.",
                priority=12
            ))
        
        # Sort by priority (highest first) and return
        recommendations.sort(key=lambda r: r.priority, reverse=True)
        return recommendations
    
    @staticmethod
    def _calculate_activity_minutes(daily_steps: int) -> int:
        """Calculate physical activity minutes from daily steps."""
        # Rough estimate: 100 steps â‰ˆ 1 minute of activity
        return min(max(daily_steps // 100, 0), 300)
    
    @staticmethod
    def _parse_activity_time(activity_time_str: str) -> int:
        """Parse physical activity time from survey response string."""
        if not activity_time_str or activity_time_str in ['Unknown', '', None, 'Not specified', 'Not Specified']:
            return None
        
        activity_time_str = activity_time_str.lower().strip()
        
        # Map common time duration responses to minutes
        # Order matters: check longer/more specific patterns first
        activity_map = {
            'more than 1 hour': 75,
            'more than 60 minutes': 75,
            '> 1 hour': 75,
            '>1 hour': 75,
            '45-60 minutes': 52,
            '45-60 min': 52,
            '45 to 60 minutes': 52,
            '30-45 minutes': 37,
            '30-45 min': 37,
            '30 to 45 minutes': 37,
            'less than 30 minutes': 20,
            'less than 30 min': 20,
            '< 30 minutes': 20,
            '<30 min': 20,
            '1 hour': 60,
            '60 minutes': 60,
            '45 minutes': 45,
            '45 min': 45,
            'none': 0,
            'no exercise': 0,
            'i don\'t exercise': 0,
        }
        
        # Map activity types to estimated minutes (if user provides activity type instead of duration)
        activity_type_map = {
            'walking': 30,
            'jogging': 45,
            'running': 60,
            'sports': 60,
            'gym': 60,
            'cycling': 45,
            'swimming': 45,
            'yoga': 30,
            'aerobics': 45,
            'dancing': 45,
            'basketball': 60,
            'football': 60,
            'soccer': 60,
            'tennis': 60,
            'hiking': 60,
        }
        
        # First check for time duration
        for key, minutes in activity_map.items():
            if key in activity_time_str:
                return minutes
        
        # Then check for activity type
        for activity_type, minutes in activity_type_map.items():
            if activity_type in activity_time_str:
                return minutes
        
        # Try to extract numeric value
        import re
        numbers = re.findall(r'\d+', activity_time_str)
        if numbers:
            return int(numbers[0])
        
        return None
    
    @staticmethod
    def _determine_activity_type(minutes: int) -> str:
        """Determine activity type based on minutes."""
        if minutes < 20:
            return "light"
        elif minutes < 60:
            return "moderate"
        else:
            return "vigorous"
    
    @staticmethod
    def format_for_api(recommendations: List[Recommendation]) -> str:
        """
        Format recommendations for API response (pipe-separated format).
        Each recommendation is formatted as "Title: Description [Source]"
        """
        return " | ".join([str(rec) for rec in recommendations])
    
    @staticmethod
    def format_for_display(recommendations: List[Recommendation], max_count: int = 10) -> List[Dict]:
        """
        Format recommendations for display (keeping only top N).
        Returns list of dicts with title, description, source.
        """
        return [
            {
                "title": rec.title,
                "description": rec.description,
                "source": rec.source
            }
            for rec in recommendations[:max_count]
        ]
