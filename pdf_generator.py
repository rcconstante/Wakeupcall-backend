from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from typing import Dict
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class WakeUpCallPDFGenerator:
    """
    Generate Sleep Apnea Report PDF with consistent layout/design
    Only values change, design stays the same
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Define custom styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=6,
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#666666'),
            spaceAfter=20
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        ))
        
        # Assessment result style (highlighted)
        self.styles.add(ParagraphStyle(
            name='HighRisk',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#c0392b'),
            fontName='Helvetica-Bold'
        ))
        
        # Explanation text style
        self.styles.add(ParagraphStyle(
            name='Explanation',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#333333'),
            spaceAfter=8,
            leading=14
        ))
        
        # Insight bullet style
        self.styles.add(ParagraphStyle(
            name='InsightBullet',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#2c3e50'),
            leftIndent=15,
            spaceAfter=4
        ))
    
    def _generate_result_explanation(self, data: Dict, risk_level: str, osa_probability: int) -> str:
        """Generate a personalized explanation of why the user received their risk assessment result"""
        
        patient = data.get('patient', {})
        stop_bang = data.get('stop_bang', {})
        ess = data.get('epworth_sleepiness_scale', {})
        lifestyle = data.get('lifestyle', {})
        medical = data.get('medical_history', {})
        
        # Extract values
        age = patient.get('age', 0)
        bmi = patient.get('bmi', 0)
        neck_cm_str = patient.get('neck_circumference', '0 cm')
        neck_cm = float(neck_cm_str.replace(' cm', '').replace('cm', '')) if isinstance(neck_cm_str, str) else neck_cm_str
        
        # Calculate STOP-BANG score from actual individual responses (not the stored score)
        stopbang_score = sum([
            1 if stop_bang.get('snoring') else 0,
            1 if stop_bang.get('tiredness') else 0,
            1 if stop_bang.get('observed_apnea') else 0,
            1 if stop_bang.get('high_blood_pressure') else 0,
            1 if stop_bang.get('bmi_over_35') else 0,
            1 if stop_bang.get('age_over_50') else 0,
            1 if stop_bang.get('neck_circumference_large') else 0,
            1 if stop_bang.get('gender_male') else 0
        ])
        
        ess_score = ess.get('total_score', 0)
        
        explanations = []
        contributing_factors = []
        
        # Risk level context
        if 'HIGH' in risk_level.upper():
            explanations.append(f"Your assessment indicates a <b>high risk</b> for obstructive sleep apnea (OSA) with an estimated probability of {osa_probability}%.")
            explanations.append("This means that based on your responses, you show multiple signs commonly associated with sleep-disordered breathing.")
        elif 'MEDIUM' in risk_level.upper() or 'MODERATE' in risk_level.upper():
            explanations.append(f"Your assessment indicates a <b>moderate risk</b> for obstructive sleep apnea (OSA) with an estimated probability of {osa_probability}%.")
            explanations.append("You show some indicators that warrant monitoring and possible further evaluation.")
        else:
            explanations.append(f"Your assessment indicates a <b>low risk</b> for obstructive sleep apnea (OSA) with an estimated probability of {osa_probability}%.")
            explanations.append("While your current risk appears low, maintaining healthy habits is important for continued good sleep health.")
        
        # Contributing factors explanation
        explanations.append("<br/><b>Key factors contributing to your result:</b>")
        
        # STOP-BANG analysis
        if stopbang_score >= 5:
            contributing_factors.append(f"• <b>High STOP-BANG Score ({stopbang_score}/8):</b> This validated screening tool identified multiple risk markers including snoring, tiredness, or observed breathing pauses.")
        elif stopbang_score >= 3:
            contributing_factors.append(f"• <b>Intermediate STOP-BANG Score ({stopbang_score}/8):</b> Some risk factors were identified that may indicate sleep apnea.")
        
        # BMI analysis
        if isinstance(bmi, (int, float)) and bmi >= 30:
            contributing_factors.append(f"• <b>Elevated BMI ({bmi:.1f}):</b> Higher body weight can increase pressure on airways during sleep, potentially causing breathing obstructions.")
        elif isinstance(bmi, (int, float)) and bmi >= 25:
            contributing_factors.append(f"• <b>Overweight BMI ({bmi:.1f}):</b> Being overweight may contribute to airway narrowing during sleep.")
        
        # Neck circumference analysis
        if neck_cm >= 40:
            contributing_factors.append(f"• <b>Large Neck Circumference ({neck_cm:.0f} cm):</b> A thicker neck often indicates more tissue that could collapse and obstruct breathing during sleep.")
        
        # Age analysis
        if isinstance(age, (int, float)) and age >= 50:
            contributing_factors.append(f"• <b>Age Factor ({age} years):</b> Sleep apnea risk increases with age as muscle tone decreases and tissue becomes more susceptible to collapse.")
        
        # ESS analysis
        if ess_score >= 11:
            contributing_factors.append(f"• <b>Elevated Daytime Sleepiness (ESS: {ess_score}/24):</b> Your responses indicate excessive daytime sleepiness, often a sign of disrupted nighttime sleep.")
        
        # Medical conditions
        if medical.get('hypertension'):
            contributing_factors.append("• <b>Hypertension:</b> High blood pressure is commonly associated with sleep apnea and may both cause and result from poor sleep quality.")
        
        if medical.get('diabetes'):
            contributing_factors.append("• <b>Diabetes:</b> Research shows a strong connection between diabetes and sleep apnea, with each condition potentially worsening the other.")
        
        # Lifestyle factors
        if lifestyle.get('smoking'):
            contributing_factors.append("• <b>Smoking:</b> Tobacco use can cause inflammation and fluid retention in the upper airway, increasing obstruction risk.")
        
        if lifestyle.get('alcohol'):
            contributing_factors.append("• <b>Alcohol Use:</b> Alcohol relaxes throat muscles, which can worsen breathing pauses during sleep.")
        
        # Combine explanations
        full_explanation = " ".join(explanations)
        if contributing_factors:
            full_explanation += "<br/>" + "<br/>".join(contributing_factors)
        
        full_explanation += "<br/><br/><i>Note: This assessment is based on validated screening tools but is not a clinical diagnosis. Consult a healthcare provider for proper evaluation and treatment recommendations.</i>"
        
        return full_explanation
    
    def generate_pdf(self, data: Dict, output_path: str = None) -> BytesIO:
        """
        Generate PDF report with fixed design, dynamic values
        
        Args:
            data: Dictionary containing all report data
            output_path: Optional file path to save PDF. If None, returns BytesIO
            
        Returns:
            BytesIO object containing the PDF
        """
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer if output_path is None else output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=50,
            bottomMargin=50
        )
        
        # Container for PDF elements
        story = []
        
        # HEADER
        story.append(Paragraph(
            "Sleep Apnea Risk Assessment – Detailed Report by<br/>WakeUpCall",
            self.styles['ReportTitle']
        ))
        
        story.append(Paragraph(
            "This detailed report includes patient information, STOP-BANG scoring, Epworth Sleepiness Scale, risk<br/>"
            "assessment, lifestyle factors, medical history, and SHAP model explanation for physician review.",
            self.styles['Subtitle']
        ))
        
        # ASSESSMENT RESULT SECTION
        story.append(Paragraph("Assessment Result", self.styles['SectionHeader']))
        
        assessment = data.get('assessment', {})
        risk_level = assessment.get('risk_level', 'N/A')
        osa_probability = assessment.get('osa_probability', 0)
        
        assessment_data = [
            ['Predicted Risk Level:', risk_level.upper()],
            ['OSA Probability:', f"{osa_probability}%"]
        ]
        
        assessment_table = Table(assessment_data, colWidths=[2*inch, 4*inch])
        assessment_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (1, 0), (1, 0), colors.HexColor('#c0392b') if 'HIGH' in risk_level.upper() else colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(assessment_table)
        story.append(Spacer(1, 0.15*inch))
        
        # WHY THIS RESULT - EXPLANATION SECTION
        story.append(Paragraph("Why This Result?", self.styles['SectionHeader']))
        
        explanation = self._generate_result_explanation(data, risk_level, osa_probability)
        story.append(Paragraph(explanation, self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # PATIENT INFORMATION SECTION
        story.append(Paragraph("Patient Information", self.styles['SectionHeader']))
        
        patient = data.get('patient', {})
        patient_data = [
            ['Name:', patient.get('name', '')],
            ['Age:', str(patient.get('age', ''))],
            ['Sex:', patient.get('sex', '')],
            ['Height:', patient.get('height', '')],
            ['Weight:', patient.get('weight', '')],
            ['BMI:', str(patient.get('bmi', ''))],
            ['Neck Circumference:', patient.get('neck_circumference', '')]
        ]
        
        patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(patient_table)
        story.append(Spacer(1, 0.2*inch))
        
        # STOP-BANG ASSESSMENT SECTION
        story.append(Paragraph("STOP-BANG Assessment", self.styles['SectionHeader']))
        
        stop_bang = data.get('stop_bang', {})
        
        # Use the stopbang_score already calculated from individual responses at the top
        actual_score = stopbang_score
        
        if actual_score >= 5:
            risk_text = "High Risk"
        elif actual_score >= 3:
            risk_text = "Intermediate Risk"
        else:
            risk_text = "Low Risk"
        
        stop_bang_data = [
            ['Snoring', 'Yes' if stop_bang.get('snoring') else 'No'],
            ['Tiredness', 'Yes' if stop_bang.get('tiredness') else 'No'],
            ['Observed Apnea', 'Yes' if stop_bang.get('observed_apnea') else 'No'],
            ['High Blood Pressure', 'Yes' if stop_bang.get('high_blood_pressure') else 'No'],
            ['BMI > 35', 'Yes' if stop_bang.get('bmi_over_35') else 'No'],
            ['Age > 50', 'Yes' if stop_bang.get('age_over_50') else 'No'],
            ['Neck ≥ 40 cm', 'Yes' if stop_bang.get('neck_circumference_large') else 'No'],
            ['Gender Male', 'Yes' if stop_bang.get('gender_male') else 'No'],
            ['Total Score', f"{actual_score}/8 ({risk_text})"]
        ]
        
        stop_bang_table = Table(stop_bang_data, colWidths=[2.5*inch, 3.5*inch])
        stop_bang_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -2), 0.5, colors.grey),
            ('LINEABOVE', (0, -1), (-1, -1), 1, colors.black),
        ]))
        
        story.append(stop_bang_table)
        story.append(Spacer(1, 0.2*inch))
        
        # EPWORTH SLEEPINESS SCALE SECTION
        story.append(Paragraph("Epworth Sleepiness Scale (ESS)", self.styles['SectionHeader']))
        
        ess = data.get('epworth_sleepiness_scale', {})
        
        # Recalculate actual ESS total from individual scores to ensure accuracy
        actual_ess_total = sum([
            ess.get('sitting_reading', 0),
            ess.get('watching_tv', 0),
            ess.get('public_sitting', 0),
            ess.get('passenger_car', 0),
            ess.get('lying_down_pm', 0),
            ess.get('talking', 0),
            ess.get('after_lunch', 0),
            ess.get('traffic_stop', 0)
        ])
        
        if actual_ess_total > 10:
            ess_interpretation = "Excessive Daytime Sleepiness"
        elif actual_ess_total > 6:
            ess_interpretation = "Higher Normal Daytime Sleepiness"
        else:
            ess_interpretation = "Normal Daytime Sleepiness"
        
        ess_data = [
            ['Sitting and reading', str(ess.get('sitting_reading', 0))],
            ['Watching TV', str(ess.get('watching_tv', 0))],
            ['Public place sitting', str(ess.get('public_sitting', 0))],
            ['Passenger in car', str(ess.get('passenger_car', 0))],
            ['Lying down PM', str(ess.get('lying_down_pm', 0))],
            ['Talking', str(ess.get('talking', 0))],
            ['After lunch', str(ess.get('after_lunch', 0))],
            ['Traffic stop', str(ess.get('traffic_stop', 0))],
            ['Total ESS Score', f"{actual_ess_total}/24 ({ess_interpretation})"]
        ]
        
        ess_table = Table(ess_data, colWidths=[2.5*inch, 3.5*inch])
        ess_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('LINEABOVE', (0, -1), (-1, -1), 2, colors.black),
        ]))
        
        story.append(ess_table)
        story.append(Spacer(1, 0.2*inch))
        
        # LIFESTYLE & MEDICAL HISTORY SECTION
        story.append(Paragraph("Lifestyle & Medical History", self.styles['SectionHeader']))
        
        lifestyle = data.get('lifestyle', {})
        medical = data.get('medical_history', {})
        
        # Get physical activity time with proper formatting
        physical_activity = lifestyle.get('physical_activity_time', 'Not specified')
        # Handle empty/unknown values
        if not physical_activity or physical_activity in ['Unknown', 'Not Specified', 'null', 'None']:
            physical_activity = 'Not specified'
        # Clean and format the value for display
        elif isinstance(physical_activity, str):
            physical_activity = physical_activity.strip()
            # Capitalize if it looks like an activity type
            if physical_activity and physical_activity[0].islower():
                physical_activity = physical_activity.capitalize()
        
        lifestyle_data = [
            ['Smoking', 'Yes' if lifestyle.get('smoking') else 'No'],
            ['Alcohol Intake', 'Yes' if lifestyle.get('alcohol') else 'No'],
            ['Physical Activity', physical_activity],
            ['Hypertension', 'Yes' if medical.get('hypertension') else 'No'],
            ['Diabetes', 'Yes' if medical.get('diabetes') else 'No']
        ]
        
        lifestyle_table = Table(lifestyle_data, colWidths=[2.5*inch, 3.5*inch])
        lifestyle_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(lifestyle_table)
        story.append(Spacer(1, 0.2*inch))
        
        # RECOMMENDATIONS SECTION
        story.append(Paragraph("Insights & Recommendations", self.styles['SectionHeader']))
        
        recommendations_text = data.get('assessment', {}).get('recommendation', '')
        if recommendations_text:
            # Split recommendations by " | " delimiter
            recommendations = recommendations_text.split(" | ")
            
            for i, rec in enumerate(recommendations[:10], 1):  # Limit to top 10
                # Parse format: "Title: Description [Source]"
                title_end = rec.find(":")
                source_start = rec.rfind("[")
                source_end = rec.rfind("]")
                
                if title_end > 0:
                    title = rec[:title_end].strip()
                    if source_start > title_end and source_end > source_start:
                        description = rec[title_end + 1:source_start].strip()
                        source = rec[source_start + 1:source_end].strip()
                    else:
                        description = rec[title_end + 1:].strip()
                        source = ""
                else:
                    title = rec[:50] if len(rec) > 50 else rec
                    description = ""
                    source = ""
                
                # Add recommendation with proper formatting
                story.append(Paragraph(
                    f"<b>{i}. {title}</b>",
                    self.styles['Normal']
                ))
                
                if description:
                    story.append(Paragraph(
                        description,
                        self.styles['Normal']
                    ))
                
                if source:
                    story.append(Paragraph(
                        f"<i>Source: {source}</i>",
                        ParagraphStyle(
                            name='SourceStyle',
                            parent=self.styles['Normal'],
                            fontSize=8,
                            textColor=colors.HexColor('#666666'),
                            leftIndent=12
                        )
                    ))
                
                story.append(Spacer(1, 0.1*inch))
        else:
            story.append(Paragraph(
                "No specific recommendations available at this time.",
                self.styles['Normal']
            ))
        
        story.append(Spacer(1, 0.2*inch))
        
        # SHAP MODEL EXPLANATION SECTION
        if 'shap_chart' in data and data['shap_chart']:
            story.append(Paragraph("SHAP Model Explanation", self.styles['SectionHeader']))
            
            story.append(Paragraph(
                "SHAP values help quantify how much each feature contributed to the final sleep apnea risk prediction. "
                "Positive values increase risk, while lower values have less influence.",
                self.styles['Normal']
            ))
            
            story.append(Spacer(1, 0.1*inch))
            
            img = Image(data['shap_chart'], width=5*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
        
        # FOOTER
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(
            f"<i>Report generated on {data.get('generated_date', '')} by WakeUpCall Sleep Health System</i>",
            self.styles['Normal']
        ))
        
        # Build PDF
        doc.build(story)
        
        if output_path is None:
            buffer.seek(0)
            return buffer
        
        return None
