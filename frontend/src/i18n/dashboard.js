export const dashboardTranslations = {
    en: {
      language: "Language",
      crop: "Select crop",
      marketForHistory: "Select market for history",
      refresh: "Refresh",
      loading: "Loading market observations...",
      loadingMarketsError: "Could not load markets",
      loadingDataError: "Could not load mandi information",
  
      eyebrow: "CROPBALANCE AI / MARKET INTELLIGENCE",
      title: "Dharashiv Mandi Dashboard",
      subtitle:
        "Explore archived crop observations, market history and comparable reported values.",
      archiveTag: "ARCHIVED DATA · NOT LIVE",
  
      cropNames: {
        soybean: "Soybean",
        onion: "Onion",
        wheat: "Wheat",
        tomato: "Tomato",
      },
  
      freshnessTitle: "DATA FRESHNESS",
      freshnessStatuses: {
        fresh: "FRESH",
        aging: "AGING",
        stale: "STALE",
        future_dated: "FUTURE DATED",
        unknown: "UNKNOWN",
      },
      freshnessMessages: {
        fresh: "These archived observations are relatively recent.",
        aging:
          "These observations are aging. Check a current market source before making a selling decision.",
        stale:
          "These observations are stale. Do not rely on them alone for a current selling decision.",
        future_dated:
          "The latest observation has a future date. Verify the archive.",
        unknown:
          "The freshness of these observations could not be established.",
      },
      latestObservation: "Latest observation",
      ageOld: (age) => `${age} day${age === 1 ? "" : "s"} old`,
      ageUnavailable: "Age unavailable",
  
      marketsCovered: "Markets covered",
      district: "Dharashiv district",
      latestObservations: "Latest observations",
      onePerMarket: "One latest observation per market",
      numericValues: "Numeric reported values",
      notVerified: "Not independently verified",
      missingInvalid: "Missing / invalid",
      nrNotZero: "NR is not treated as zero",
  
      marketOverview: "MARKET OVERVIEW",
      latestCropObservations: (crop) => `Latest ${crop} observations`,
      districtLocation: "Dharashiv, Maharashtra",
      notReported: "Not reported",
      unavailable: "Unavailable",
      unitUnavailable: "Unit unavailable",
      unspecifiedVariety: "Unspecified variety",
      viewHistory: "View history",
  
      historicalObservations: "HISTORICAL OBSERVATIONS",
      selectedMarket: "Selected market",
      records: (count) => `${count} records`,
      noNumericHistory:
        "No numeric historical observations are available for this market.",
      chartCaption:
        "Values are archived AGMARKNET reports. Missing observations remain missing; the line does not represent a live quote.",
      chartReportedValue: "Reported value",
  
      comparableObservations: "COMPARABLE OBSERVATIONS",
      marketComparison: "Market comparison",
      comparisonGroups: (count) => `${count} comparison groups`,
      markets: (count) => `${count} markets`,
      highestReported: "Highest reported value",
      noComparisons:
        "There are not enough matching observations to compare markets.",
      dataCaveat:
        "A higher reported value does not establish the best place to sell. Confirm today's price, crop grade, quantity, transport costs and market fees before making a selling decision.",
  
      footer: "CropBalance AI · Local Mandi Intelligence · Dharashiv pilot",
    },
  
    hi: {
      language: "भाषा",
      crop: "फसल चुनें",
      marketForHistory: "इतिहास देखने के लिए मंडी चुनें",
      refresh: "रीफ़्रेश करें",
      loading: "मंडी के आंकड़े लोड हो रहे हैं...",
      loadingMarketsError: "मंडियों की सूची लोड नहीं हो सकी",
      loadingDataError: "मंडी की जानकारी लोड नहीं हो सकी",
  
      eyebrow: "क्रॉपबैलेंस AI / मंडी जानकारी",
      title: "धाराशिव मंडी डैशबोर्ड",
      subtitle:
        "संग्रहित फसल जानकारी, मंडी का इतिहास और मंडियों के रिपोर्ट किए गए मूल्यों की तुलना करें।",
      archiveTag: "संग्रहित डेटा · लाइव नहीं",
  
      cropNames: {
        soybean: "सोयाबीन",
        onion: "प्याज़",
        wheat: "गेहूँ",
        tomato: "टमाटर",
      },
  
      freshnessTitle: "डेटा की ताज़गी",
      freshnessStatuses: {
        fresh: "नया",
        aging: "पुराना होता हुआ",
        stale: "बहुत पुराना",
        future_dated: "भविष्य की तारीख",
        unknown: "अज्ञात",
      },
      freshnessMessages: {
        fresh: "यह संग्रहित जानकारी अपेक्षाकृत नई है।",
        aging:
          "यह जानकारी पुरानी हो रही है। फसल बेचने का निर्णय लेने से पहले वर्तमान मंडी भाव की जाँच करें।",
        stale:
          "यह जानकारी बहुत पुरानी है। आज की बिक्री का निर्णय केवल इस जानकारी के आधार पर न लें।",
        future_dated:
          "नवीनतम जानकारी की तारीख भविष्य में है। संग्रह की जाँच करें।",
        unknown: "इस जानकारी की ताज़गी की पुष्टि नहीं हो सकी।",
      },
      latestObservation: "नवीनतम जानकारी",
      ageOld: (age) => `${age} दिन पुरानी`,
      ageUnavailable: "उम्र उपलब्ध नहीं",
  
      marketsCovered: "शामिल मंडियाँ",
      district: "धाराशिव जिला",
      latestObservations: "नवीनतम जानकारी",
      onePerMarket: "हर मंडी की नवीनतम जानकारी",
      numericValues: "संख्यात्मक रिपोर्ट किए गए मूल्य",
      notVerified: "स्वतंत्र रूप से सत्यापित नहीं",
      missingInvalid: "गायब / अमान्य",
      nrNotZero: "NR को शून्य नहीं माना जाता",
  
      marketOverview: "मंडी की जानकारी",
      latestCropObservations: (crop) => `${crop} की नवीनतम जानकारी`,
      districtLocation: "धाराशिव, महाराष्ट्र",
      notReported: "रिपोर्ट नहीं हुआ",
      unavailable: "उपलब्ध नहीं",
      unitUnavailable: "इकाई उपलब्ध नहीं",
      unspecifiedVariety: "किस्म निर्दिष्ट नहीं",
      viewHistory: "पुराना रिकॉर्ड देखें",
  
      historicalObservations: "पुराने बाजार रिकॉर्ड",
      selectedMarket: "चुनी गई मंडी",
      records: (count) => `${count} रिकॉर्ड`,
      noNumericHistory:
        "इस मंडी के लिए कोई संख्यात्मक पुराना रिकॉर्ड उपलब्ध नहीं है।",
      chartCaption:
        "ये AGMARKNET के संग्रहित रिकॉर्ड हैं। गायब जानकारी को गायब ही रखा गया है। यह लाइव भाव नहीं है।",
      chartReportedValue: "रिपोर्ट किया गया मूल्य",
  
      comparableObservations: "तुलना योग्य रिकॉर्ड",
      marketComparison: "मंडी की तुलना",
      comparisonGroups: (count) => `${count} तुलना समूह`,
      markets: (count) => `${count} मंडियाँ`,
      highestReported: "सबसे अधिक रिपोर्ट किया गया मूल्य",
      noComparisons:
        "मंडियों की तुलना के लिए पर्याप्त मिलते-जुलते रिकॉर्ड उपलब्ध नहीं हैं।",
      dataCaveat:
        "अधिक रिपोर्ट किया गया मूल्य यह साबित नहीं करता कि वही बेचने के लिए सबसे अच्छी मंडी है। बिक्री का निर्णय लेने से पहले आज का भाव, फसल की गुणवत्ता, मात्रा, परिवहन खर्च और मंडी शुल्क जाँचें।",
  
      footer: "CropBalance AI · स्थानीय मंडी जानकारी · धाराशिव पायलट",
    },
  
    mr: {
      language: "भाषा",
      crop: "पीक निवडा",
      marketForHistory: "इतिहास पाहण्यासाठी बाजार निवडा",
      refresh: "रिफ्रेश करा",
      loading: "बाजाराची माहिती लोड होत आहे...",
      loadingMarketsError: "बाजारांची यादी लोड करता आली नाही",
      loadingDataError: "बाजाराची माहिती लोड करता आली नाही",
  
      eyebrow: "क्रॉपबॅलन्स AI / बाजार माहिती",
      title: "धाराशिव बाजार डॅशबोर्ड",
      subtitle:
        "संग्रहित पीक नोंदी, बाजाराचा इतिहास आणि नोंदवलेल्या मूल्यांची तुलना करा.",
      archiveTag: "संग्रहित माहिती · थेट नाही",
  
      cropNames: {
        soybean: "सोयाबीन",
        onion: "कांदा",
        wheat: "गहू",
        tomato: "टोमॅटो",
      },
  
      freshnessTitle: "माहितीची अद्ययावत स्थिती",
      freshnessStatuses: {
        fresh: "ताजी",
        aging: "जुनी होत आहे",
        stale: "खूप जुनी",
        future_dated: "भविष्यातील तारीख",
        unknown: "अज्ञात",
      },
      freshnessMessages: {
        fresh: "ही संग्रहित माहिती तुलनेने अलीकडची आहे.",
        aging:
          "ही माहिती जुनी होत आहे. विक्रीचा निर्णय घेण्यापूर्वी सध्याचा बाजारभाव तपासा.",
        stale:
          "ही माहिती खूप जुनी आहे. आजच्या विक्रीचा निर्णय फक्त या माहितीवर घेऊ नका.",
        future_dated:
          "सर्वात अलीकडील नोंदीची तारीख भविष्यातील आहे. संग्रह तपासा.",
        unknown: "या माहितीची ताजेपणा निश्चित करता आली नाही.",
      },
      latestObservation: "शेवटची नोंद",
      ageOld: (age) => `${age} दिवस जुनी`,
      ageUnavailable: "वय उपलब्ध नाही",
  
      marketsCovered: "समाविष्ट बाजार",
      district: "धाराशिव जिल्हा",
      latestObservations: "नवीनतम नोंदी",
      onePerMarket: "प्रत्येक बाजाराची नवीनतम नोंद",
      numericValues: "संख्यात्मक नोंदवलेली मूल्ये",
      notVerified: "स्वतंत्रपणे पडताळलेले नाही",
      missingInvalid: "गहाळ / अवैध",
      nrNotZero: "NR ला शून्य मानले जात नाही",
  
      marketOverview: "बाजाराचा आढावा",
      latestCropObservations: (crop) => `${crop} च्या नवीनतम नोंदी`,
      districtLocation: "धाराशिव, महाराष्ट्र",
      notReported: "नोंदवलेले नाही",
      unavailable: "उपलब्ध नाही",
      unitUnavailable: "एकक उपलब्ध नाही",
      unspecifiedVariety: "जात नमूद केलेली नाही",
      viewHistory: "इतिहास पाहा",
  
      historicalObservations: "ऐतिहासिक नोंदी",
      selectedMarket: "निवडलेला बाजार",
      records: (count) => `${count} नोंदी`,
      noNumericHistory:
        "या बाजारासाठी संख्यात्मक ऐतिहासिक नोंदी उपलब्ध नाहीत.",
      chartCaption:
        "ही AGMARKNET ची संग्रहित माहिती आहे. गहाळ नोंदी गहाळच ठेवल्या आहेत. ही थेट बाजारभावाची माहिती नाही.",
      chartReportedValue: "नोंदवलेले मूल्य",
  
      comparableObservations: "तुलना करण्यायोग्य नोंदी",
      marketComparison: "बाजारांची तुलना",
      comparisonGroups: (count) => `${count} तुलना गट`,
      markets: (count) => `${count} बाजार`,
      highestReported: "सर्वाधिक नोंदवलेले मूल्य",
      noComparisons:
        "बाजारांची तुलना करण्यासाठी पुरेशा जुळणाऱ्या नोंदी उपलब्ध नाहीत.",
      dataCaveat:
        "जास्त नोंदवलेले मूल्य म्हणजे तोच विक्रीसाठी सर्वोत्तम बाजार असे सिद्ध होत नाही. विक्रीचा निर्णय घेण्यापूर्वी आजचा भाव, पिकाचा दर्जा, प्रमाण, वाहतूक खर्च आणि बाजार शुल्क तपासा.",
  
      footer: "CropBalance AI · स्थानिक बाजार माहिती · धाराशिव पायलट",
    },
  };