/**
 * TranslationButton Component
 *
 * A button component that allows users to translate chapter content to Urdu.
 * This component handles the translation functionality and UI.
 */

import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import styles from './TranslationButton.module.css';

const TranslationButton = ({ contentId, onTranslate, currentLanguage = 'en' }) => {
  const [isTranslating, setIsTranslating] = useState(false);
  const [translationStatus, setTranslationStatus] = useState('idle'); // idle, loading, success, error

  const handleTranslate = async () => {
    if (currentLanguage === 'ur') {
      // If already in Urdu, switch back to English
      onTranslate && onTranslate('en');
      return;
    }

    setIsTranslating(true);
    setTranslationStatus('loading');

    try {
      // In a real implementation, this would call the backend translation API
      // For now, we'll simulate the translation process
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API call

      // Call the onTranslate callback with 'ur' for Urdu
      onTranslate && onTranslate('ur');
      setTranslationStatus('success');

      // Reset status after a short delay
      setTimeout(() => setTranslationStatus('idle'), 2000);
    } catch (error) {
      console.error('Translation failed:', error);
      setTranslationStatus('error');
      setTimeout(() => setTranslationStatus('idle'), 3000);
    } finally {
      setIsTranslating(false);
    }
  };

  // Determine button text based on current language
  const getButtonText = () => {
    if (translationStatus === 'loading') {
      return 'Translating...';
    }
    if (currentLanguage === 'ur') {
      return 'انگریزی میں تبدیل کریں'; // Switch to English
    }
    return 'اردو میں ترجمہ کریں'; // Translate to Urdu
  };

  // Determine button class based on status
  const getButtonClass = () => {
    return clsx(
      styles.translateButton,
      {
        [styles.loading]: translationStatus === 'loading',
        [styles.success]: translationStatus === 'success',
        [styles.error]: translationStatus === 'error',
        [styles.urduMode]: currentLanguage === 'ur',
      }
    );
  };

  return (
    <div className={styles.translationContainer}>
      <button
        className={getButtonClass()}
        onClick={handleTranslate}
        disabled={isTranslating}
        title={currentLanguage === 'ur' ? 'Switch to English' : 'Translate to Urdu'}
      >
        <span className={styles.buttonText}>
          {getButtonText()}
        </span>
        {translationStatus === 'loading' && (
          <span className={styles.spinner}>-spinner-</span>
        )}
        {translationStatus === 'success' && (
          <span className={styles.checkmark}>✓</span>
        )}
        {translationStatus === 'error' && (
          <span className={styles.errorIcon}>⚠️</span>
        )}
      </button>
      {translationStatus === 'error' && (
        <div className={styles.errorMessage}>
          Translation failed. Please try again.
        </div>
      )}
    </div>
  );
};

export default TranslationButton;